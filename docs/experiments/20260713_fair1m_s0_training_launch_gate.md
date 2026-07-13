# FAIR1M S0 Training Launch Gate (2026-07-13)

## Outcome

The FAIR1M S0 campaign was not launched. The mandatory data gate failed on
two active zero-area train records, and the first GPU preflight observed a
transient NVIDIA driver communication failure. A later three-poll check
recovered, but the data failure remains sufficient to stop the campaign.

## Completed checks

- Official checkpoint downloaded from
  `https://download.pytorch.org/models/resnet50-0676ba61.pth`.
- Size: `102530333` bytes.
- SHA-256:
  `0676ba61b6795bbe1773cffd859882e5e297624d384b6993f7c9e683e722fb8a`.
- PyTorch CPU load succeeded (`OrderedDict`, 267 entries).
- Full train scan: `208927` image/annotation pairs and `1785001` active
  objects; exact stems, no unknown classes, and MMRotate conversion errors
  were absent.
- Full `ss_val` scan: `10970` pairs and `199347` active objects; exact stems,
  no malformed records, no unknown classes, and zero invalid MMRotate rboxes.
- Config load, dataloader construction, and one complete two-sample batch
  passed on CPU. The batch contained 3 and 6 ground-truth instances.
- Config checks passed: 37 canonical hyphenated classes, sanitized paths,
  batch size 2, 12 epochs, validation/checkpoint interval 4, seed 3407, and
  the checked local ResNet-50 path.
- Canonical RemoteCLIP metadata loaded with shape `[37, 512]` and class order
  `a220` through `warship`.

## Failed gates

The train scan found these two zero-area active records:

```text
train/annfiles/16088__533__0___0.txt:3
train/annfiles/16088__800__0___0.txt:3
```

Both contain a finite, known `other-airplane` polygon whose post-tile
coordinates have area `0.0`. The scan reported `num_malformed_records: 2`.
It also reported 93 out-of-bounds objects among the 300 representative train
images decoded; these require review before reopening the zero-malformed gate.

The initial `nvidia-smi` preflight failed with “couldn't communicate with the
NVIDIA driver.” A later three-poll record recovered, but GPUs 1, 2, and 6
were occupied throughout; GPUs 0, 3, 4, and 5 remained at 14 MiB and 0%.
No FAIR1M process was started.

## Not run

- Exactly-1,000-batch train-step diagnostic.
- One-epoch smoke, validation metric, or checkpoint acceptance.
- Three detached 12-epoch replicas.

Replica config metadata was added for seeds `3407`, `4407`, and `5407` under
`OpenRSD/M_configs/G02_Baselines/Data3_FAIR1M/`, but no replica workdir,
screen, PID, GPU assignment, or checkpoint was created.

Reports:

- `New/artifacts/fair1m_geometry_gate_20260713_mmrotate.json`
- `New/artifacts/fair1m_geometry_gate_20260713_mmrotate.md`
- `OpenRSD/work_dirs/geonexus_fair1m/fair1m_config_gate_20260713.json`
