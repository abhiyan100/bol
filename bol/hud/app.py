"""The pill window. Run as `python -m bol.hud.app`, fed JSON lines on stdin.

Everything AppKit lives here, in a child process, so the daemon can stay a
background process with no window and no app identity. The window itself is
built to be invisible to the rest of the system: an accessory activation
policy (no Dock tile, no menu bar), a non-activating borderless panel that is
only ever ordered front (never made key), and mouse events passed straight
through. Focus stays exactly where the user left it, which is the whole point:
Bol pastes into the frontmost terminal.

Reads stdin on a thread, applies on the main thread, exits when stdin closes.
"""

from __future__ import annotations

import argparse
import sys
import threading

from .render import Update, hold_for, parse_line, render, truncate_middle

# Pill geometry, in points.
HEIGHT = 30.0
PAD_X = 14.0
DOT = 8.0
GAP = 8.0
RADIUS = 12.0
FONT_SIZE = 13.0
MIN_WIDTH = 88.0
MAX_SCREEN_FRACTION = 0.6
TOP_INSET = 24.0
BOTTOM_INSET = 40.0
FADE_IN_S = 0.12
FADE_OUT_S = 0.20
# The 13 pt system font averages a little under this per character; it only
# has to be close, the text cell truncates whatever is left over.
CHAR_WIDTH = 7.0

# sRGB dot colours, keyed by the names render.color_for returns.
DOT_COLORS = {
    "green": (0.30, 0.85, 0.39),
    "blue": (0.35, 0.62, 1.00),
    "amber": (1.00, 0.72, 0.23),
    "red": (1.00, 0.32, 0.28),
}


class Pill:
    """One borderless panel that shows one line at a time."""

    def __init__(self, position: str = "top") -> None:
        import AppKit

        try:
            # Only for the type registration: without it PyObjC hands back an
            # untyped pointer for CGColor and warns on every dot repaint.
            import Quartz  # noqa: F401
        except Exception:
            pass

        self.AppKit = AppKit
        self.position = position if position in ("top", "bottom") else "top"
        self._visible = False
        self._timer = None

        rect = AppKit.NSMakeRect(0.0, 0.0, 240.0, HEIGHT)
        style = (
            AppKit.NSWindowStyleMaskBorderless
            | AppKit.NSWindowStyleMaskNonactivatingPanel
        )
        panel = AppKit.NSPanel.alloc().initWithContentRect_styleMask_backing_defer_(
            rect, style, AppKit.NSBackingStoreBuffered, False
        )
        panel.setOpaque_(False)
        panel.setBackgroundColor_(AppKit.NSColor.clearColor())
        panel.setHasShadow_(True)
        panel.setLevel_(AppKit.NSStatusWindowLevel)
        panel.setIgnoresMouseEvents_(True)
        panel.setHidesOnDeactivate_(False)
        panel.setFloatingPanel_(True)
        panel.setBecomesKeyOnlyIfNeeded_(True)
        panel.setCollectionBehavior_(
            AppKit.NSWindowCollectionBehaviorCanJoinAllSpaces
            | AppKit.NSWindowCollectionBehaviorFullScreenAuxiliary
            | AppKit.NSWindowCollectionBehaviorStationary
            | AppKit.NSWindowCollectionBehaviorIgnoresCycle
        )
        panel.setAppearance_(
            AppKit.NSAppearance.appearanceNamed_(AppKit.NSAppearanceNameVibrantDark)
        )
        panel.setAlphaValue_(0.0)

        blur = AppKit.NSVisualEffectView.alloc().initWithFrame_(rect)
        blur.setMaterial_(AppKit.NSVisualEffectMaterialHUDWindow)
        blur.setBlendingMode_(AppKit.NSVisualEffectBlendingModeBehindWindow)
        blur.setState_(AppKit.NSVisualEffectStateActive)
        blur.setWantsLayer_(True)
        blur.layer().setCornerRadius_(RADIUS)
        blur.layer().setMasksToBounds_(True)
        panel.setContentView_(blur)

        dot = AppKit.NSView.alloc().initWithFrame_(
            AppKit.NSMakeRect(PAD_X, (HEIGHT - DOT) / 2.0, DOT, DOT)
        )
        dot.setWantsLayer_(True)
        dot.layer().setCornerRadius_(DOT / 2.0)
        blur.addSubview_(dot)

        label = AppKit.NSTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(PAD_X, 0.0, 100.0, HEIGHT)
        )
        label.setBezeled_(False)
        label.setDrawsBackground_(False)
        label.setEditable_(False)
        label.setSelectable_(False)
        label.setFont_(AppKit.NSFont.systemFontOfSize_(FONT_SIZE))
        label.setTextColor_(AppKit.NSColor.whiteColor())
        label.setUsesSingleLineMode_(True)
        label.cell().setLineBreakMode_(AppKit.NSLineBreakByTruncatingMiddle)
        label.cell().setScrollable_(False)
        blur.addSubview_(label)

        self.panel = panel
        self.dot = dot
        self.label = label
        self.font = label.font()

    # ------------------------------------------------------------------ apply

    def apply(self, update: Update) -> None:
        """Show one update. Called on the main thread only."""
        self._cancel_timer()
        label, color = render(update)
        if not label:
            self.hide()
            return
        self.show(label, color)
        hold = hold_for(update.state)
        if hold:
            self._timer = self.AppKit.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
                hold, False, lambda _timer: self.hide()
            )

    def show(self, text: str, color: str) -> None:
        self._layout(text, color)
        if self._visible:
            return
        self._visible = True
        self.panel.setAlphaValue_(0.0)
        # Never makeKeyAndOrderFront_: that would steal focus from the
        # terminal Bol is about to paste into.
        self.panel.orderFrontRegardless()
        self._fade(1.0, FADE_IN_S)

    def hide(self) -> None:
        self._cancel_timer()
        if not self._visible:
            return
        self._visible = False
        self._fade(0.0, FADE_OUT_S, self._order_out)

    def _order_out(self) -> None:
        if not self._visible:
            self.panel.orderOut_(None)

    # ----------------------------------------------------------------- layout

    def _layout(self, text: str, color: str) -> None:
        AppKit = self.AppKit
        screen = self._screen()
        area = screen.visibleFrame()
        max_width = area.size.width * MAX_SCREEN_FRACTION
        lead = PAD_X + (DOT + GAP if color else 0.0)
        room = max(CHAR_WIDTH, max_width - lead - PAD_X)
        text = truncate_middle(text, max(8, int(room / CHAR_WIDTH)))

        self.label.setStringValue_(text)
        # sizeToFit is the only measurement that agrees with the cell's own
        # insets; a bare string measurement is a couple of points short, and
        # the cell answers by truncating a label that would have fitted.
        self.label.sizeToFit()
        measured = self.label.frame().size.width
        width = min(max(MIN_WIDTH, lead + measured + PAD_X), max_width)

        if color:
            self.dot.setHidden_(False)
            red, green, blue = DOT_COLORS.get(color, DOT_COLORS["blue"])
            self.dot.layer().setBackgroundColor_(
                AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(
                    red, green, blue, 1.0
                ).CGColor()
            )
        else:
            self.dot.setHidden_(True)

        self.label.setFrame_(
            AppKit.NSMakeRect(lead, 0.0, max(1.0, width - lead - PAD_X), HEIGHT)
        )
        x = area.origin.x + (area.size.width - width) / 2.0
        if self.position == "bottom":
            y = screen.frame().origin.y + BOTTOM_INSET
        else:
            y = area.origin.y + area.size.height - HEIGHT - TOP_INSET
        self.panel.setFrame_display_(AppKit.NSMakeRect(x, y, width, HEIGHT), True)

    def _screen(self):
        """The screen the user is looking at.

        An accessory app has no key window, so it cannot ask where the
        frontmost window is. The cursor is the practical proxy, and it is
        where the user's attention is anyway.
        """
        AppKit = self.AppKit
        point = AppKit.NSEvent.mouseLocation()
        for screen in AppKit.NSScreen.screens():
            if AppKit.NSPointInRect(point, screen.frame()):
                return screen
        return AppKit.NSScreen.mainScreen()

    # ---------------------------------------------------------------- effects

    def _fade(self, alpha: float, seconds: float, then=None) -> None:
        AppKit = self.AppKit
        AppKit.NSAnimationContext.beginGrouping()
        context = AppKit.NSAnimationContext.currentContext()
        context.setDuration_(seconds)
        if then is not None:
            context.setCompletionHandler_(then)
        self.panel.animator().setAlphaValue_(alpha)
        AppKit.NSAnimationContext.endGrouping()

    def _cancel_timer(self) -> None:
        timer, self._timer = self._timer, None
        if timer is not None:
            timer.invalidate()


def pump(stream, apply, done) -> None:
    """Read JSON lines until stdin closes, then ask the app to quit."""
    try:
        while True:
            line = stream.readline()
            if not line:
                break
            update = parse_line(line)
            if update is not None:
                apply(update)
    finally:
        done()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="bol.hud.app", description="Bol's status pill.")
    parser.add_argument("--position", default="top", choices=("top", "bottom"))
    args = parser.parse_args(argv)

    try:
        import AppKit
        from PyObjCTools import AppHelper
    except Exception as exc:  # PyObjC missing or no window server
        sys.stderr.write(f"bol.hud: AppKit is not available ({exc}), no pill this run.\n")
        return 0

    app = AppKit.NSApplication.sharedApplication()
    # Accessory: no Dock tile, no menu bar, and it can never become the
    # frontmost application.
    app.setActivationPolicy_(AppKit.NSApplicationActivationPolicyAccessory)
    pill = Pill(args.position)

    def apply(update):
        AppHelper.callAfter(pill.apply, update)

    def done():
        AppHelper.callAfter(AppHelper.stopEventLoop)

    reader = threading.Thread(
        target=pump, args=(sys.stdin, apply, done), name="bol-hud-stdin", daemon=True
    )
    reader.start()
    AppHelper.runEventLoop(installInterrupt=True)
    return 0


if __name__ == "__main__":  # pragma: no cover - process entry point
    raise SystemExit(main())
