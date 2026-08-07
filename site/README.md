# mlx-cv landing page

Static GitHub Pages site for [mlx-cv](https://github.com/appautomaton/mlx-cv),
published at <https://appautomaton.renocrypt.com/mlx-cv/>.

## Editing rule

**Nothing on this page carries a number that has to be maintained.** Parity
measurements, test counts, tensor counts, and checkpoint layouts all live in the
repository, where they are produced. The page states what does not change — the
tasks, the loader contract, the parity discipline, the scope — and links out for
anything measured. When editing, grep the page for digits: if a figure would go
stale after a gate run, it belongs in the README instead.

## Stack

A single self-contained `index.html` — no build step, no framework. Styles and
the small amount of JS (theme toggle, mobile menu, scroll reveal) are inline.

- **Type:** Bespoke Stencil (display), Switzer (body), Martian Mono (data).
  Bespoke Stencil is never set below ~28px; the cut letterforms are a
  segmentation mask, which is the point, but they need size to read.
- **Colour:** a three-stop ramp — indigo, teal, amber — used one hue per
  section, so scrolling the page walks a colormap. Hue order is fixed and
  lightness flips between themes so every accent stays legible as text and as a
  1.5px icon stroke.
- **Theme:** light by default. Light/dark via `data-theme` on `<html>`,
  persisted in localStorage, falling back to the OS scheme. Deep-link with
  `?theme=light` / `?theme=dark`.
- **Icons:** hand-drawn SVG, no icon library. Each stroke carries
  `pathLength="100"` so a single CSS rule can trace all of them regardless of
  their real length. Two rules here are easy to break by accident:
  - The dash rules select `.d`, not `[pathLength]`. Blink does not invalidate a
    camelCase attribute selector on an SVG child when a class lands on an
    ancestor, so selecting the attribute leaves every icon undrawn. A new
    stroke needs `class="d"` as well as `pathLength="100"`.
  - A filled shape carries its resting opacity in `--fill-o`, because a CSS
    `opacity` outranks the SVG presentation attribute and would otherwise
    flatten a translucent mask into a solid disc.
- **Motion:** [Motion](https://motion.dev) 13.0.0 from jsDelivr, pinned. It only
  drives scroll reveals; the hero animation is pure CSS.
- **Responsive:** authored mobile-first; breakpoints at 720px and 940px.
- `prefers-reduced-motion` is respected — every animation renders in its final
  state.

The signature element is the headline: it is drawn twice, once as an outlined
"reference" layer and once solid, and the offset between them closes on load.
That is the library's premise rendered as type.

## Deploy

Published by `.github/workflows/pages.yml` on every push to `main` that touches
`site/`. There is no build: the workflow uploads this directory as-is. The Pages
source is set to **GitHub Actions**. `.nojekyll` keeps Jekyll out of the way.

## Local preview

```bash
python3 -m http.server -d site 8000
# open http://localhost:8000/
```

## The share card

`assets/og.png` is 2400×1260 (1.91:1 at 2×), rendered from `assets/og.html`
with the real webfonts rather than approximated. The source is committed so the
card can be re-cut whenever the headline or the task list changes:

1. open `site/assets/og.html` at a 1200×630 viewport, `deviceScaleFactor` 2;
2. wait for `document.fonts.ready` — screenshotting early bakes in the fallback
   face, which is the whole reason this is a real browser render;
3. capture the viewport to `site/assets/og.png`.

`og.html` carries `noindex` and is not in the sitemap; it deploys with the rest
of the directory only because the workflow uploads `site/` as-is.
