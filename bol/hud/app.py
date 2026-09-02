"""The pill window. Run as `python -m bol.hud.app`, fed JSON lines on stdin.

Everything AppKit lives here, in a child process, so the daemon can stay a
background process with no window and no app identity. The window itself is
built to be invisible to the rest of the system: an accessory activation
policy (no Dock tile, no menu bar), a non-activating borderless panel that is
only ever ordered front (never made key), and mouse events passed straight
through. Focus stays exactly where the user left it, which is the whole point:
Bol pastes into the frontmost terminal.

What it draws is a fixed-width dark capsule: Bol's mark on the left, five
dots on the right, and nothing else unless [ui] text is on. Fixed width
matters more than it sounds: a pill that resized itself for every state
twitched at the top of the screen all day. The dots carry the state instead,
and render.py owns which pattern belongs to which state.

Reads stdin on a thread, applies on the main thread, exits when stdin closes.
"""

from __future__ import annotations

import argparse
import sys
import threading
import time

from .render import (
    DOT_COUNT,
    Update,
    animated,
    dot_alphas,
    dots_for,
    draft_span,
    hold_for,
    label_for,
    parse_line,
    truncate_middle,
)

# Capsule geometry, in points. Every one of these is a design decision the
# reference sketch made; changing one without the others makes the pill look
# like a control rather than a badge.
HEIGHT = 44.0
RADIUS = HEIGHT / 2.0  # a full capsule, not a rounded box
PAD_X = 12.0
ICON = 26.0
ICON_RADIUS = 7.0
ICON_GAP = 12.0
GLYPH_PT = 14.0
DOT = 9.0
DOT_GAP = 8.0
TEXT_GAP = 12.0
FONT_SIZE = 13.0

DOTS_X = PAD_X + ICON + ICON_GAP
DOTS_W = DOT_COUNT * DOT + (DOT_COUNT - 1) * DOT_GAP
# The pill's whole width with no text: mark, gap, five dots, padding.
WIDTH = DOTS_X + DOTS_W + PAD_X

MAX_SCREEN_FRACTION = 0.6
TOP_INSET = 24.0
BOTTOM_INSET = 40.0
FADE_IN_S = 0.12
FADE_OUT_S = 0.20
# The 13 pt system font averages a little under this per character; it only
# has to be close, the text cell truncates whatever is left over.
CHAR_WIDTH = 7.0
# Words the decoder has not committed yet are drawn at this alpha, so the
# sentence visibly settles from the left instead of rewriting itself.
DRAFT_ALPHA = 0.6

# How often an animated pattern repaints. Fast enough that a sweep looks
# like motion, slow enough that a decoration is never the reason a fan spins.
FRAME_S = 1.0 / 30.0

# The capsule itself: near-black over the blur, with a hairline of white so
# it keeps an edge against a dark terminal behind it.
SCRIM = (0.04, 0.04, 0.05, 0.85)
BORDER = (1.0, 1.0, 1.0, 0.08)
# The mark: a light rounded square with a dark glyph in it.
ICON_FILL = (1.0, 1.0, 1.0, 0.97)
GLYPH_INK = (0.06, 0.06, 0.07, 1.0)

# sRGB dot colours, keyed by the names render.dots_for returns. "" is the
# plain white dot, which is most of them.
DOT_COLORS = {
    "": (1.00, 1.00, 1.00),
    "green": (0.30, 0.85, 0.39),
    "blue": (0.45, 0.68, 1.00),
    "amber": (1.00, 0.72, 0.23),
    "red": (1.00, 0.35, 0.31),
}


def _utf16_len(text: str) -> int:
    """Length in UTF-16 units, which is what an NSRange is measured in."""
    return len(text.encode("utf-16-le", "surrogatepass")) // 2


class Pill:
    """One borderless capsule: a mark, five dots, and an optional line."""

    def __init__(self, position: str = "top", text: bool = False) -> None:
        import AppKit

        try:
            # Only for the type registration: without it PyObjC hands back an
            # untyped pointer for CGColor and warns on every repaint.
            import Quartz  # noqa: F401
        except Exception:
            pass

        self.AppKit = AppKit
        self.position = position if position in ("top", "bottom") else "top"
        self.text = bool(text)
        self._visible = False
        self._hold_timer = None
        self._anim_timer = None
        self._dots = dots_for("idle")
        self._level = 0.0
        self._started = time.monotonic()
        self._frame = None

        rect = AppKit.NSMakeRect(0.0, 0.0, WIDTH, HEIGHT)
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

        # The blur alone is too pale over a light window; the scrim is what
        # makes the capsule read as one dark object wherever it lands.
        scrim = AppKit.NSView.alloc().initWithFrame_(rect)
        scrim.setWantsLayer_(True)
        scrim.setAutoresizingMask_(
            AppKit.NSViewWidthSizable | AppKit.NSViewHeightSizable
        )
        layer = scrim.layer()
        layer.setCornerRadius_(RADIUS)
        layer.setBackgroundColor_(self._cg(*SCRIM))
        layer.setBorderWidth_(1.0)
        layer.setBorderColor_(self._cg(*BORDER))
        blur.addSubview_(scrim)

        self.panel = panel
        self.blur = blur
        self.scrim = scrim
        self.icon = self._build_icon()
        self.dots = self._build_dots()
        self.label = self._build_label()
        self.font = self.label.font()

    # ----------------------------------------------------------------- build

    def _cg(self, red: float, green: float, blue: float, alpha: float):
        return self.AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(
            red, green, blue, alpha
        ).CGColor()

    def _build_icon(self):
        """The mark: a light rounded square with a waveform in it.

        A placeholder for a real logo, and drawn as one view so a later asset
        is a one-line swap. The glyph is optional on purpose: SF Symbols
        arrived in macOS 11, and an older Mac should get a blank mark rather
        than no pill.
        """
        AppKit = self.AppKit
        frame = AppKit.NSMakeRect(PAD_X, (HEIGHT - ICON) / 2.0, ICON, ICON)
        view = AppKit.NSView.alloc().initWithFrame_(frame)
        view.setWantsLayer_(True)
        view.layer().setCornerRadius_(ICON_RADIUS)
        view.layer().setBackgroundColor_(self._cg(*ICON_FILL))
        self.blur.addSubview_(view)

        image = self._symbol("waveform")
        if image is not None:
            glyph = AppKit.NSImageView.alloc().initWithFrame_(
                AppKit.NSMakeRect(0.0, 0.0, ICON, ICON)
            )
            glyph.setImage_(image)
            glyph.setImageScaling_(AppKit.NSImageScaleProportionallyDown)
            try:
                glyph.setContentTintColor_(
                    AppKit.NSColor.colorWithSRGBRed_green_blue_alpha_(*GLYPH_INK)
                )
            except Exception:  # noqa: BLE001 - a tint is not worth a crash
                pass
            view.addSubview_(glyph)
        return view

    def _symbol(self, name: str):
        """One SF Symbol at the mark's weight, or None on an older macOS."""
        AppKit = self.AppKit
        try:
            image = AppKit.NSImage.imageWithSystemSymbolName_accessibilityDescription_(
                name, None
            )
        except Exception:  # noqa: BLE001 - macOS 10.15 and older
            return None
        if image is None:
            return None
        try:
            config = AppKit.NSImageSymbolConfiguration.configurationWithPointSize_weight_(
                GLYPH_PT, AppKit.NSFontWeightSemibold
            )
            image = image.imageWithSymbolConfiguration_(config) or image
        except Exception:  # noqa: BLE001 - unconfigured is still a glyph
            pass
        return image

    def _build_dots(self):
        AppKit = self.AppKit
        y = (HEIGHT - DOT) / 2.0
        views = []
        for index in range(DOT_COUNT):
            x = DOTS_X + index * (DOT + DOT_GAP)
            view = AppKit.NSView.alloc().initWithFrame_(
                AppKit.NSMakeRect(x, y, DOT, DOT)
            )
            view.setWantsLayer_(True)
            view.layer().setCornerRadius_(DOT / 2.0)
            self.blur.addSubview_(view)
            views.append(view)
        return views

    def _build_label(self):
        AppKit = self.AppKit
        label = AppKit.NSTextField.alloc().initWithFrame_(
            AppKit.NSMakeRect(self._text_x(), 0.0, 1.0, HEIGHT)
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
        label.setHidden_(not self.text)
        self.blur.addSubview_(label)
        return label

    def _text_x(self) -> float:
        return DOTS_X + DOTS_W + TEXT_GAP

    # ------------------------------------------------------------------ apply

    def apply(self, update: Update) -> None:
        """Show one update. Called on the main thread only."""
        self._cancel_hold()
        self._cancel_anim()
        dots = dots_for(update.state, update.hold)
        if dots.motion == "hidden":
            self.hide()
            return

        self._dots = dots
        self._level = update.level
        self._started = time.monotonic()
        label = label_for(update.state, update.text, update.detail) if self.text else ""
        draft = update.detail if update.state == "listening" else ""
        self._layout(label, draft)
        self._paint()
        self.show()

        if animated(dots):
            self._anim_timer = self._every(FRAME_S, self._paint)
        hold = hold_for(update.state, update.hold)
        if hold:
            self._hold_timer = self._after(hold, self.hide)

    def show(self) -> None:
        if self._visible:
            return
        self._visible = True
        self.panel.setAlphaValue_(0.0)
        # Never makeKeyAndOrderFront_: that would steal focus from the
        # terminal Bol is about to paste into.
        self.panel.orderFrontRegardless()
        self._fade(1.0, FADE_IN_S)

    def hide(self) -> None:
        self._cancel_hold()
        self._cancel_anim()
        if not self._visible:
            return
        self._visible = False
        self._fade(0.0, FADE_OUT_S, self._order_out)

    def _order_out(self) -> None:
        if not self._visible:
            self.panel.orderOut_(None)

    # ------------------------------------------------------------------ paint

    def _paint(self, _timer=None) -> None:
        """Repaint the dots and the mark for wherever the motion is now."""
        elapsed = time.monotonic() - self._started
        alphas = dot_alphas(self._dots, elapsed, self._level)
        red, green, blue = DOT_COLORS.get(self._dots.color, DOT_COLORS[""])
        for view, alpha in zip(self.dots, alphas):
            view.layer().setBackgroundColor_(self._cg(red, green, blue, alpha))
        self.icon.setAlphaValue_(self._dots.icon)

    # ----------------------------------------------------------------- layout

    def _layout(self, text: str, draft: str = "") -> None:
        """Place the panel. The width is fixed unless there is text to show."""
        AppKit = self.AppKit
        screen = self._screen()
        area = screen.visibleFrame()
        width = WIDTH
        if self.text and text:
            max_width = max(WIDTH, area.size.width * MAX_SCREEN_FRACTION)
            room = max(CHAR_WIDTH, max_width - self._text_x() - PAD_X)
            text = truncate_middle(text, max(8, int(room / CHAR_WIDTH)))
            # Measured after truncation, so a draft the panel had to cut is
            # drawn solid rather than dimmed in the wrong place.
            dim = draft_span("listening", text, draft) if draft else 0
            self.label.setAttributedStringValue_(self._attributed(text, dim))
            # sizeToFit is the only measurement that agrees with the cell's
            # own insets; a bare string measurement is a couple of points
            # short, and the cell answers by truncating a label that fitted.
            self.label.sizeToFit()
            measured = self.label.frame().size
            width = min(self._text_x() + measured.width + PAD_X, max_width)
            self.label.setHidden_(False)
            # Centred on the line the dots sit on, not on the cell: a text
            # field given the capsule's full height draws its one line at the
            # top of it, which at 44 pt reads as a caption that slipped.
            self.label.setFrame_(
                AppKit.NSMakeRect(
                    self._text_x(),
                    (HEIGHT - measured.height) / 2.0,
                    max(1.0, width - self._text_x() - PAD_X),
                    measured.height,
                )
            )
        else:
            self.label.setHidden_(True)

        x = area.origin.x + (area.size.width - width) / 2.0
        if self.position == "bottom":
            y = screen.frame().origin.y + BOTTOM_INSET
        else:
            y = area.origin.y + area.size.height - HEIGHT - TOP_INSET
        frame = (x, y, width, HEIGHT)
        # The listening meter re-applies fifteen times a second and almost
        # never changes the geometry; moving the window anyway makes it
        # shimmer.
        if frame == self._frame:
            return
        self._frame = frame
        self.panel.setFrame_display_(AppKit.NSMakeRect(x, y, width, HEIGHT), True)

    def _attributed(self, text: str, dim: int):
        """The label's string, with the last `dim` characters faded out."""
        AppKit = self.AppKit
        white = AppKit.NSColor.whiteColor()
        string = AppKit.NSMutableAttributedString.alloc().initWithString_attributes_(
            text,
            {
                AppKit.NSFontAttributeName: self.font,
                AppKit.NSForegroundColorAttributeName: white,
            },
        )
        if dim:
            # NSRange counts UTF-16 units and Python counts code points, so an
            # emoji anywhere in the line would otherwise put the range past the
            # end of the string and take the window down.
            total, tail = _utf16_len(text), _utf16_len(text[-dim:])
            try:
                string.addAttribute_value_range_(
                    AppKit.NSForegroundColorAttributeName,
                    white.colorWithAlphaComponent_(DRAFT_ALPHA),
                    (total - tail, tail),
                )
            except Exception as exc:  # noqa: BLE001 - dim text is decoration
                sys.stderr.write(f"bol.hud: could not dim the draft ({exc})\n")
        return string

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

    def _after(self, seconds: float, call):
        return self.AppKit.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
            seconds, False, lambda _timer: call()
        )

    def _every(self, seconds: float, call):
        return self.AppKit.NSTimer.scheduledTimerWithTimeInterval_repeats_block_(
            seconds, True, lambda _timer: call()
        )

    def _cancel_hold(self) -> None:
        timer, self._hold_timer = self._hold_timer, None
        if timer is not None:
            timer.invalidate()

    def _cancel_anim(self) -> None:
        timer, self._anim_timer = self._anim_timer, None
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
    parser.add_argument(
        "--text",
        action="store_true",
        help="show the current line beside the dots ([ui] text).",
    )
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
    pill = Pill(args.position, args.text)

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
