# what-each-probe-measures

Each probe measures a perturbation sensitivity diagnostic, not model quality.

- `position_bias`: swaps A/B response order and checks whether the semantic preference changes.
- `verbosity_bias`: compares a concise weak answer to a neutrally padded version of the same answer.
- `self_preference`: reports preference rate when one side is labeled with the judge backend's model family.
- `paraphrase_stability`: scores five synthetic paraphrases of the same answer and reports preference changes plus score variance.
- `anchoring`: measures first-position preference over paired original/swapped trials.
- `calibration`: compares judge confidence to synthetic quality labels and reports expected calibration error plus reliability bins.
