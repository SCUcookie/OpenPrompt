# Dataset And Resource Manifest

This file should be filled as soon as the real assets are known.

## Codebase

Current status:

- unknown

Target fields:

```text
Codebase path:
Repository name:
Branch:
Main training entrypoint:
Main evaluation entrypoint:
OpenRSD implementation present: yes / no / unknown
```

## Hardware

Current status:

- only the upper constraint is known: at most 4 x 4090

Target fields:

```text
GPU count:
GPU model:
VRAM per GPU:
CPU:
RAM:
Free storage:
CUDA version:
```

## Datasets

Fill one block per dataset.

### Template

```text
Dataset:
Available on disk: yes / no
Absolute path:
Task type: OBB / HBB / segmentation / mixed
Train split ready: yes / no
Val split ready: yes / no
Test split ready: yes / no
Annotation format:
License or usage restriction:
Notes:
```

### Priority datasets

Use this order when filling the manifest:

1. DOTA-v2.0
2. DIOR-R
3. FAIR1M-2.0
4. HRSC2016
5. SpaceNet
6. VEDAI
7. CORS
8. DOSR
9. SODA-A

## Prompt resources

Target fields:

```text
Text prompt source:
Image prompt source:
Hierarchy prompt source:
Scene prior source:
LLM available for offline hierarchy generation:
VLM encoder available:
```

## Resource truthfulness rule

Do not fill a field from memory if it was not verified on disk or provided by the user.
