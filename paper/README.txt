Current manuscript: main.tex and included sections; compiled main.pdf.
Main text occupies four pages through Conclusion. Limitations and references start on page 5; ethics and references follow on pages 5–6; appendices start on page 7; total twelve pages.
Build: python build_paper.py. Uses installed pdfLaTeX and BibTeX, including local MiKTeX discovery on Windows.

Figure 1: two-column-wide teaser at top of page 2. Editable source: figures/teaser_editable_v2.pptx. Text, shapes, arrows, and chart are editable; chart values are included in an embedded workbook. Actual PowerPoint open/export verified. figures/teaser.pdf is the vector PowerPoint export used in LaTeX.
Concept: figures/teaser_concept.png, generated with the built-in image tool. Exact prompt and generation method: figures/teaser_concept_prompt.txt. The concept is not the final figure: distillation routing, fixed language execution, chart data and baseline were corrected during redraw.
PowerPoint builder: make_teaser.mjs, using the Codex bundled @oai/artifact-tool. It records local runtime paths and runs from the top-level workspace; place it in tmp/teaser_pptx with a node_modules link to the bundled runtime. Finalizer output and receipt names must be new when rerunning. export_teaser_powerpoint.ps1 uses installed PowerPoint to export the finalized file. These tools are for figure authoring only.
Figure 2: python make_figures.py creates the intervention loss plot from saved CSV results. It no longer overwrites the teaser.
Figure 3: python make_discourse_figure.py creates selected speech-act paths from the saved discourse manifest. Both Python scripts require repository dependencies and make no model calls.
Table 1: main comparisons. Tables 2–6 and six discourse episodes: appendices. Appendix D and its Figures 4–5 cover assignment-required diagnostics from existing saved results. Rebuild them with python make_assignment_diagnostics.py.

English TeX is the latest manuscript. Earlier bilingual files retain prior wording; limitations_bilingual.txt remains current for the shortened limitations.
ACL style, fonts, and margins are unchanged. Prior figures/drafts: drafts/pre_layout_20261012.
Verification: layout_validation.json and teaser_pptx_validation.json. No new negotiation experiments or paid negotiation model calls were made during this layout revision.

Overleaf upload: ../submission/overleaf_upload.zip. Editable unpacked project: ../submission/overleaf_source/. Author information: authors.tex; actual team record: ../submission/TEAM_RECORD.txt.
