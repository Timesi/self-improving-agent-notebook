# papers/ directory

Each lecture has a subdirectory `lecture-XX/` with paper PDFs and study notes:

- **NOTES.md** — Chinese study notes (in version control). Each note covers: core ideas, key formulas/algorithms,
  key experimental numbers, a teaching thread (how a Stanford instructor might teach this), code demo ideas,
  exercise ideas, and references. Notebooks are written from these notes.
- **NOTES.en.md** — English sibling of the same notes. Same headings, structure, lists, formulas, and links.
- **\*.pdf** — original papers (not in version control, about 150MB). Re-download with:

```bash
python scripts/download_papers.py           # download all
python scripts/download_papers.py --lecture 02   # one lecture
python scripts/download_papers.py --dry-run      # plan only
```

- **Template**: `NOTES_TEMPLATE.md` / `NOTES_TEMPLATE.en.md` define the study-note structure.

## Notes

- Some lectures have no arXiv paper (AlphaGeometry appeared in Nature, AlphaProof is a DeepMind blog,
  guest lectures have no assigned papers). Those materials are stored as `.md` files
  (for example `lecture-13/alphageometry-readme.md` and `alphageometry-readme.en.md`).
