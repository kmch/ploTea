# Figure requirements that shaped this package

Every default in plotea is answerable to a journal's figure requirements rather than to taste,
and this page is the source those defaults were read off. It is quoted rather than paraphrased,
because the wording is the specification: *one point* is a number a line width has to meet, and
"avoid perceptually non-uniform colour scales" is why no rainbow colormap is offered.

Where plotea implements one of these, it is named in brackets.

## Nature Geoscience

Source: https://www.nature.com/ngeo/for-authors/preparing-your-submission
Figures should be numbered separately with Arabic numerals in the order of occurrence in the text of the manuscript. One- or two-column format figures are preferred. When appropriate, figures should include error bars. A description of the statistical treatment of error analysis should be included in the figure or scheme legend.

Figure lettering should be in a clear, **sans-serif** typeface (for example, `Helvetica`); if possible, the same typeface in approximately the **same font size** should be used for all figures in a paper. Use symbol font for Greek letters. **All display items should be on a white background**, and should avoid excessive boxing, unnecessary colour, spurious decorative effects (such as three-dimensional 'skyscraper' histograms) and highly pixelated computer drawings. The vertical axis of histograms should not be truncated to exaggerate small differences. Labelling must be of sufficient size and contrast to be readable, even after appropriate reduction. The **thinnest lines in the final figure should be no smaller than one point wide**. Reasonable requests to enlarge figures will be considered, but editors will make the final decision on figure size. Authors will see a proof of figures.

Figures divided into parts should be labelled with a **lower-case bold a, b, and so on, in the same type size as used elsewhere in the figure**. Lettering in figures should be in lower-case type, with **only the first letter of each label capitalized**. **Units should have a single space between the number and the unit**, and follow SI nomenclature (for example, ms rather than msec) or the nomenclature common to a particular field. **Thousands should be separated by commas (1,000)**. Unusual units or abbreviations should be spelled out in full or defined in the legend. Scale bars should be used rather than magnification factors, with the length of the bar defined in the legend rather than on the bar itself. In legends, please use visual cues rather than verbal explanations, such as "open red triangles".

Unnecessary figures should be avoided: data presented in small tables or histograms, for instance, can generally be stated briefly in the text instead. **Figures should not contain more than one panel unless the parts are logically connected**; each panel of a multipart figure should be sized so that the whole figure can be reduced by the same amount and reproduced on the printed page at the smallest size at which essential details are visible.

Authors are encouraged to consider the needs of colour-blind readers (a substantial minority of the male population) when choosing colours for figures. Many colour-blind readers cannot interpret visuals that rely on discrimination of green and red, for example. The **use of colour-safe combinations**, such as green and magenta, turquoise and red, yellow and blue or other accessible colour palettes is strongly encouraged. Please also **avoid the use of perceptually non-uniform colour scales, such as rainbows**. Further reading: The misuse of colour in science communication.

When a manuscript is accepted for publication, we will ask for high-resolution figure files, possibly in a different electronic format. This information will be included in the acceptance letter. See below for details of digital image production and submission.

## How plotea answers them

Verified against the code rather than assumed, because a page like this is worth nothing if it
describes intentions.

| the requirement | what plotea does |
| --- | --- |
| sans-serif lettering, one typeface throughout | `styles/plotea.mplstyle` sets `font.family: Arial` |
| thinnest lines no smaller than one point | the same file sets `axes.linewidth: 1` and `lines.linewidth: 1`, against matplotlib's defaults of 0.8 and 1.5 |
| lower-case bold **a**, **b** panel labels | {func}`~plotea.generic.labels.panel_label`, bold by default |
| scale bars rather than magnification factors, length defined in the legend | {func}`~plotea.generic.scalebar.scalebar` draws a segmented bar in projected coordinates and leaves the number to the caption |
| white background, no spurious decoration | the basemap styles in {mod}`plotea.maps.styles` fill land and sea flat and draw no frame furniture |
| labelling readable and consistent across panels | {meth}`~plotea.maps.colorbar.Colorbar.format_ticks` fixes the formatter once, so one panel cannot read `0.05` while its neighbour reads `5e-02` |
| one panel unless the parts are logically connected, all reduced by the same amount | {mod}`plotea.maps.layout` aligns equal-aspect map axes so a figure scales as a whole |

Two requirements the package does **not** currently meet, recorded here rather than quietly
omitted. It offers no colour-blind-safe sequential ramp: {mod}`plotea.maps.cmaps` carries
discrete class schemes (ESA WorldCover, Koeppen-Geiger) and a land-terrain ramp, and
matplotlib's `terrain`, which the latter builds on, is itself perceptually non-uniform -- so the
rule against non-uniform scales is satisfied only by there being no rainbow on offer, not by the
default being uniform. And nothing formats thousands with commas; `format_ticks` governs
scientific notation, not digit grouping.

These are Nature Geoscience's rules, and most Nature-family journals differ only in detail. The
point of the table is that the defaults already aim at this class of requirement, so a
submission elsewhere means adjusting rather than starting over.
