# Report — ACL Style (7–8 pages)

## Templates (from spec §5.2)

- LaTeX: https://www.overleaf.com/latex/templates/association-for-computational-linguistics-acl-conference/jvxskxpnznfj
- Word: https://2023.aclweb.org/downloads/acl2023.docx
- Sample paper: https://aclanthology.org/2025.acl-long.5.pdf

## Constraints

- 7–8 pages **excluding** references and appendix
- Turnitin threshold: 15% (plagiarism + AI); enrollment key in spec §7
- Title + author information required
- Citations in ACL style (see `acl.bst` in the template)

## Required sections → where the content comes from

| Report section | Source |
|---|---|
| Abstract | Notebook §11 summary + notebook section 0 |
| Introduction (motivation, dataset overview, research questions) | Notebook §1–2 |
| Related Work | Dataset paper (Zhang et al. 2015, char-CNN) + topic-classification prior work + model papers (BERT, Word2Vec, GloVe, …) |
| Methodology — Dataset + EDA | Notebook §2–3 (figures: class distribution, length stats, word clouds) |
| Methodology — Preprocessing + justification | Notebook §4 (strategy comparison results) |
| Methodology — Word representations | Notebook §6 |
| Methodology — Model architectures + hyperparameters | Notebook §7–10 (tuning tables) |
| Results (consolidated table, confusion heatmaps, best/worst discussion) | Notebook §11 |
| Conclusion (takeaways, limitations, future work) | Notebook §12 |

## Notes

- Export notebook figures at high DPI and keep originals in `data/processed/figures/`.
- The 15% Turnitin threshold is strict — write prose in your own words; quote sparingly.
