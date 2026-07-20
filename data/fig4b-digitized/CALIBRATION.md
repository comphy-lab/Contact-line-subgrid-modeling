# Figure 4(b) vector-extraction calibration

Source: Snoeijer & Andreotti, Annu. Rev. Fluid Mech. 45:269-292 (2013),
PDF page index 5 (printed page 274), `page.get_drawings()` via PyMuPDF.
All coordinates below are PDF device coordinates (points, origin top-left,
y increasing downward; page rect 531 x 657).

## Axis calibration (pure vector, no OCR guesswork)

x-axis (Ca x 10^3): tick lines in drawing #3 at
x = 315.740 (label "2"), 351.207 ("4"), 386.595 ("6"), 421.990 ("8"),
457.457 ("10"). Perfectly linear: sx = (457.457 - 315.740)/8 = 17.7147 pt
per unit of Ca x 10^3; origin x0 = 280.311 pt (matches the "0" label
centred at x = 280.3 and the frame left edge at 280.35).

    Ca = (x_dev - 280.311) / 17.7147 / 1000        [absolute Ca]

y-axis (z/l_gamma): tick lines in drawing #1 at
y = 243.816, 276.709, 309.678, 342.647, 375.616, 408.503 — six ticks
0.5 apart with labels 3, 2, 1, 0 on the major ones. sy = (408.503 -
243.816)/2.5 = 65.875 pt per unit z; z = 0 extrapolates to y = 441.44,
matching the bottom frame line (441.22-441.72) and the "0" label centre.

    z/l_gamma = 3.0 - (y_dev - 243.816) / 65.875

Frame spans Ca x 10^3 = 0 to 11.32, z/l_gamma = 0 to 3.51.

## Objects extracted

- Theory curve: drawing #157 — the only thick stroke (width 1.5 pt),
  gray RGB ~(0.345, 0.348, 0.356), 7 cubic Bezier segments. Each segment
  sampled at 60 points -> 421-point ordered polyline (ordered bottom to
  top: bath end first). A thin dark twin of the same path (drawing #7,
  width 0.5) was ignored as a styling duplicate.
- Dashed vertical lines: drawings #4 and #5, dashed gray strokes at
  x = 441.42 and 477.22 -> Ca = 9.095e-3 and 11.116e-3.
- Symbols (marker centre = bounding-box centre of each small (<8 pt)
  vector glyph, deduplicated within 1.5 pt, sorted by increasing z):
  - red    open circles   (stroke 0.924,0.173,0.180): 27 pts
  - green  filled circles (fill 0.000,0.666,0.309):   26 pts
  - yellow open squares   (stroke 0.973,0.800,0.046):  9 pts
  - magenta filled squares(fill 0.809,0.152,0.566):   11 pts
  - blue   open triangles (stroke 0.000,0.251,0.444):  6 pts
  White-fill under-layers of the open symbols and the thin per-colour
  connecting polylines (drawings #152-#156, width 0.4) were excluded.

## Ambiguities / caveats

- The red and green series are also drawn as continuous thin wiggly
  experimental traces (drawings #152, #153); only their overlaid marker
  glyphs are exported as "symbols". The full traces were not exported.
- One tiny dense yellow scribble path (drawing #113, 133 items in a
  2x3 pt box centred at Ca = 10.65e-3, z = 1.25) and a similar dark one
  (#145 at Ca = 9.68e-3, z = 1.14) could not be classified as clean
  marker glyphs and were left out (each would be at most one extra
  point where yellow squares already cluster).
- Marker centres carry ~0.5 pt positional uncertainty
  (~3e-5 in Ca, ~0.008 in z).
- No raster fallback was needed; everything is vector.

## Verification (against the task's expected features)

- Curve start: (Ca ~ 6e-6, z = 0.627) ~ (0, 0.65)  PASS
- Fold (max Ca): 10.54e-3 at z = 1.446 (expected 10.5-11.3e-3, z 1.3-1.5)  PASS
- Upper branch end at frame top: Ca = 9.294e-3 at z = 3.51, approaching
  the dashed asymptote at 9.095e-3  PASS
- 400-dpi render (fig4b_render.png) visually matches the extracted
  geometry (gray fold left of the right dashed line, symbol clusters).
