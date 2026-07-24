---
name: paper
description: Search for ONE new research paper relevant to the GeoNexus-RSD project and write a method-focused introduction to a new dated markdown file. Use this skill whenever the user asks to find a paper, find related work, look for a new/recent paper, search literature, find a comparison/comparator method, or asks "find me a paper on X" — for remote sensing detection/segmentation, vision-language prompting, oriented object detection, or adjacent fields. Also use it when the user wants recent SOTA to compare against or to inform future work.
---

# Find-One-Paper (GeoNexus-RSD)

Deliver exactly **one** paper per invocation, chosen by a fixed priority
order, written up as a method-focused markdown note. The value is not the
citation — it is a self-contained method explanation the user can act on
without opening the PDF, plus an honest route-impact judgment. The
2026-07-23 ZODS-RS note
(`docs/literature/20260723_zods_rs_method_introduction.md`) is the
reference template; match its depth and honesty.

## Selection priority (apply strictly, in order)

When multiple candidates surface, rank them by these criteria in order —
a later criterion only breaks ties among candidates that tie on the earlier
one. State in the note why the chosen paper won on this ranking.

1. **Correlation with our project.** Closest first: oriented remote sensing
   detection, vision-language / prompt / hierarchy / context methods,
   DIOR-R / DOTA / FAIR1M benchmarks, then detection-segmentation
   intersection, then adjacent aerial-vision work. A tightly-related paper
   at a weaker venue beats a loosely-related paper at a top venue.
2. **Recency and publish status.** Prefer the most recent, and among recent
   ones prefer an **accepted/published** record (proceedings, journal,
   OpenReview accept) over an arXiv-only preprint. Verify the venue from a
   primary page — never a GitHub README badge (that has been wrong twice
   in this project). If the best-correlated paper is arXiv-only, that is
   allowed, but say so plainly and flag it as not-yet-a-comparator.
3. **Open-source code.** Prefer papers with a released, reachable code
   repository. Confirm the repo actually exists and is non-empty (a
   promised "code coming soon" does not count as open-source). Note the
   repo URL and whether checkpoints/configs are present.
4. **Usefulness for our future work, ideally as a comparison.** Favor
   papers we could realistically evaluate against or borrow a mechanism
   from — a method that reports DIOR-R/DOTA/FAIR1M numbers under a
   matchable protocol, or whose idea slots into TPC/HRR/SCA, outranks a
   method that is interesting but unusable here.
5. **One paper only.** Never deliver a list. If two are genuinely worthy,
   pick one and mention the runner-up in a single closing line.

## Search method

- Use WebSearch broadly, then WebFetch the primary source (arXiv abstract
  page, CVF/OpenReview/journal page) to confirm details. When arXiv HTML
  is available (`arxiv.org/html/<id>`), fetch it for the real method
  content — abstracts are not enough for a method write-up.
- Consult `docs/literature/20260607_openrsd_related_recent_papers.md` first
  to avoid re-picking an already-tracked paper, and to see what angles are
  already covered.
- Verify venue/acceptance and code from primary pages. If a claimed venue
  or repo cannot be confirmed, record it as unverified rather than
  asserting it.

## Output file

Write to `docs/literature/<YYYYMMDD>_<shortname>_method_introduction.md`
(matching the ZODS-RS note's location and naming). Required sections:

1. **Header block**: exact title, all authors, source URL, submission/
   publication date, verified venue/status, code repo URL + presence check,
   and a one-line "what it is".
2. **Why this paper, given our progress**: the ranking justification —
   which priority criteria it won on, and why it beat the alternatives.
3. **Problem setting**: the gap the paper addresses, in our vocabulary.
4. **Method in detail** (the core, longest section): the full pipeline /
   architecture in order, the key equations with their meaning, and the
   design insight behind each component. Explain *why* each piece exists,
   not just what it is — the user should understand the method well enough
   to critique it.
5. **Results**: datasets, metrics, baselines, headline numbers — with an
   explicit note on whether they are protocol-comparable to our DIOR-R /
   DOTA-v2.0 / FAIR1M numbers (usually they are NOT; say so, and never
   imply a false head-to-head).
6. **Stated limitations**: what the authors themselves concede.
7. **Relevance to GeoNexus-RSD (route impact)**: concrete, honest — is it a
   usable comparator, a mechanism to borrow into TPC/HRR/SCA, a future-work
   direction, or reference-only? Respect the standing route closures
   (FAIR1M S2, DIOR-R S4, segmentation lane, DOTA-v2.0 follow-up training):
   a paper does not reopen a closed route; note if it would only be
   actionable as analysis-only work.

## After writing

Add a short dated entry to the literature tracker
(`docs/literature/20260607_openrsd_related_recent_papers.md`) pointing at
the new note, so the pick is recorded and not re-searched next time.
Report to the user with the paper's identity, why it won the ranking, the
2-3 sentence method essence, and the single most actionable route-impact
line.

## Honesty guardrails

- arXiv-only is not a top-venue record; a README badge is not venue
  evidence; "code coming soon" is not open-source. State each plainly.
- Never present the paper's numbers as comparable to ours unless the split,
  metric, and evaluator genuinely match — they rarely do.
- If nothing genuinely relevant and recent turns up, say so rather than
  padding with a weak pick.
