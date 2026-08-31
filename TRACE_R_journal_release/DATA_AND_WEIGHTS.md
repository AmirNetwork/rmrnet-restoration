# Data and weights

Datasets and third-party checkpoints are not redistributed in the code archive.
They retain their original licenses.

## Controlled road data

- IVCNZ pothole images: prepare source-separated train, validation, and test
  manifests before rendering corruptions.
- PCM road-damage data: Zenodo DOI `10.5281/zenodo.17834373`; preserve pothole,
  crack, and manhole labels and group the same source frame before splitting.
- The released builders produce one 82-field observable packet per rendered
  image. Hidden blur kernels and scenario labels are not inference inputs.

## CRID

CRID comprises 4,134 native 4752x3168 Sony ILX-LR1 frames and synchronized SBG
records. The reported field study uses 320 human-reviewed frames with 565 boxes
and a frozen temporal 180/60/80 adaptation/validation/test split. The project
repository is the release location for de-identified images, synchronized
telemetry, annotations, and alignment summaries. Publication of unredacted
street imagery and precise coordinates remains subject to privacy review
because frames may contain people, vehicle plates, and location information.
The present code archive therefore contains hashes and protocol manifests, not
unredacted images or precise coordinates.

## Checkpoints

- TRACE-R selected checkpoint SHA-256:
  `a79e2a775e576f17cfe78688484985830e89de7fbe582eca10d43cc4e0cf59db`
- IVCNZ detector SHA-256:
  `7d7e24e4e13e85456578b505dcf7ba327ab923d1fbd68fc2127e04766c96b4b9`
- PCM detector SHA-256:
  `7b6db99cd29da5ed4488d99a0afce2606491222251ce2669592290133562d290`
- CRID detector SHA-256:
  `2dab19dca6809ad45372317937fd83453c7b1ffc97da871912b02b4a62f115fc`
- CRID TRACE-R field-policy checkpoint SHA-256:
  `f67019ef510b47f6a21f9036e6f5a3f74479154f54288e3937c76d174b3ff045`

Download DeMoE, DFPIR, and InstructIR weights from their official repositories.
The NAFNet comparison starts from the authors' released
`NAFNet-GoPro-width32.pth` checkpoint. Its expected SHA-256 is
`19394e6155d12ef6371d1d57496f87f0ec88f92bdffa27c0792690722d5d1a5c`.
The paper bibliography gives the corresponding publications.
