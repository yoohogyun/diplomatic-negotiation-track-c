Minimal Overleaf collaboration project

Upload
- New project: choose New Project > Upload Project in Overleaf and select overleaf_upload.zip.
- Existing shared project: upload the contents of overleaf_source to the project root, preserving the figures subdirectory and replacing old files with the same names.
- Main document: main.tex. Compiler: pdfLaTeX. Bibliography: BibTeX (processed automatically).
- Include all figures/*.pdf files. The compiled main.pdf, experiment code, raw logs, checkpoints, .env, and PPTX are not needed for compilation.

Fields to complete
1. authors.tex: actual names, affiliations, and student IDs. Use English text with pdfLaTeX.
2. main.tex, line 2: change [review] to [final] for named course submission. Keep review for an anonymous draft.
3. Record actual roles and teammate execution separately in TEAM_RECORD.txt in the submission directory.
4. After editing authors or content, verify that the end of Limitations remains within four pages. References follow without a forced page break. Preserve the original ACL fonts and margins.

Editing map
main.tex: title, abstract, Introduction, Related Work
methods_results.tex: Method, Experimental Setup, Results, main discourse analysis
discussion_conclusion.tex: Discussion, Conclusion
limitations.tex: limitations and disclosure of model assistance; no separate Ethics section
appendix_experiments.tex: first appendix page, core environment/training details and six discourse cases
appendix_assignment.tex: second appendix page, transitions, learning curves, metrics, interventions, and sensitivity
assignment_*_table.tex: tables generated from saved results
references.bib: references
figures/: five vector PDF figures used for compilation

To edit the teaser, open the separately supplied teaser_editable_v2.pptx in PowerPoint, export to PDF, and replace figures/teaser.pdf.
Check any numerical edits against the saved experiment results.
Current layout: Limitations ends at the bottom of page 4; References flow naturally onto pages 5–6; appendices occupy pages 7–8. Total: eight pages and 24 cited works.
The full-width teaser appears below the title/authors and above the abstract. The paperteaser macro in main.tex adds it to the ACL title block.
Extended supplementary details remain in paper/supplementary_appendix_full.tex in the GitHub repository.

Official instructions
https://www.overleaf.com/learn/latex/Kb/Uploading_a_project
https://docs.overleaf.com/managing-projects-and-files/adding-files-to-a-project/uploading-files-to-a-project
https://docs.overleaf.com/getting-started/recompiling-your-project/the-main-document
