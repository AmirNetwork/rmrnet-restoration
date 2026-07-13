# GT46 YOLO26 Coordinate Evaluation

| Run | Images | GT | Pred | GT success | P@0.10 | R@0.10 | F1@0.10 | F1@0.50 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| raw | 44 | 167 | 96 | 0.311 | 0.510 | 0.293 | 0.373 | 0.274 |
| rmr_blind | 44 | 167 | 92 | 0.293 | 0.511 | 0.281 | 0.363 | 0.247 |
| rmr_metadata | 44 | 167 | 95 | 0.299 | 0.505 | 0.287 | 0.366 | 0.260 |
| rmr_metadata_gated | 44 | 167 | 98 | 0.317 | 0.510 | 0.299 | 0.377 | 0.264 |
| rmr_detdom2ep_metadata_eta0p05 | 44 | 167 | 98 | 0.323 | 0.520 | 0.305 | 0.385 | 0.272 |
| rmr_native_gate_gamma085 | 44 | 167 | 101 | 0.335 | 0.525 | 0.317 | 0.396 | 0.276 |
| rmr_dual_evidence | 44 | 167 | 137 | 0.449 | 0.511 | 0.419 | 0.461 | 0.283 |
| nafnet | 44 | 167 | 110 | 0.269 | 0.382 | 0.251 | 0.303 | 0.217 |
| dfpir | 44 | 167 | 112 | 0.251 | 0.366 | 0.246 | 0.294 | 0.201 |
| demoe_auto | 44 | 167 | 99 | 0.305 | 0.495 | 0.293 | 0.368 | 0.278 |
| demoe_scenario | 44 | 167 | 99 | 0.287 | 0.455 | 0.269 | 0.338 | 0.248 |
| instructir_generic | 44 | 167 | 96 | 0.317 | 0.510 | 0.293 | 0.373 | 0.266 |
| instructir_metadata | 44 | 167 | 105 | 0.281 | 0.419 | 0.263 | 0.324 | 0.221 |

## Per-Class Exact-Class Metrics at IoU 0.10

| Run | Class | GT | Pred | TP | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| raw | Alligator Crack | 32 | 15 | 10 | 0.667 | 0.312 | 0.426 |
| raw | Longitudinal Crack | 76 | 23 | 17 | 0.739 | 0.224 | 0.343 |
| raw | Potholes | 27 | 44 | 14 | 0.318 | 0.519 | 0.394 |
| raw | Transverse Crack | 32 | 14 | 4 | 0.286 | 0.125 | 0.174 |
| rmr_blind | Alligator Crack | 32 | 18 | 11 | 0.611 | 0.344 | 0.440 |
| rmr_blind | Longitudinal Crack | 76 | 23 | 15 | 0.652 | 0.197 | 0.303 |
| rmr_blind | Potholes | 27 | 44 | 14 | 0.318 | 0.519 | 0.394 |
| rmr_blind | Transverse Crack | 32 | 7 | 3 | 0.429 | 0.094 | 0.154 |
| rmr_metadata | Alligator Crack | 32 | 18 | 12 | 0.667 | 0.375 | 0.480 |
| rmr_metadata | Longitudinal Crack | 76 | 22 | 15 | 0.682 | 0.197 | 0.306 |
| rmr_metadata | Potholes | 27 | 43 | 13 | 0.302 | 0.481 | 0.371 |
| rmr_metadata | Transverse Crack | 32 | 12 | 4 | 0.333 | 0.125 | 0.182 |
| rmr_metadata_gated | Alligator Crack | 32 | 15 | 10 | 0.667 | 0.312 | 0.426 |
| rmr_metadata_gated | Longitudinal Crack | 76 | 27 | 19 | 0.704 | 0.250 | 0.369 |
| rmr_metadata_gated | Potholes | 27 | 45 | 14 | 0.311 | 0.519 | 0.389 |
| rmr_metadata_gated | Transverse Crack | 32 | 11 | 3 | 0.273 | 0.094 | 0.140 |
| rmr_detdom2ep_metadata_eta0p05 | Alligator Crack | 32 | 15 | 10 | 0.667 | 0.312 | 0.426 |
| rmr_detdom2ep_metadata_eta0p05 | Longitudinal Crack | 76 | 25 | 18 | 0.720 | 0.237 | 0.356 |
| rmr_detdom2ep_metadata_eta0p05 | Potholes | 27 | 45 | 15 | 0.333 | 0.556 | 0.417 |
| rmr_detdom2ep_metadata_eta0p05 | Transverse Crack | 32 | 13 | 4 | 0.308 | 0.125 | 0.178 |
| rmr_native_gate_gamma085 | Alligator Crack | 32 | 14 | 10 | 0.714 | 0.312 | 0.435 |
| rmr_native_gate_gamma085 | Longitudinal Crack | 76 | 30 | 21 | 0.700 | 0.276 | 0.396 |
| rmr_native_gate_gamma085 | Potholes | 27 | 46 | 14 | 0.304 | 0.519 | 0.384 |
| rmr_native_gate_gamma085 | Transverse Crack | 32 | 11 | 4 | 0.364 | 0.125 | 0.186 |
| rmr_dual_evidence | Alligator Crack | 32 | 20 | 14 | 0.700 | 0.438 | 0.538 |
| rmr_dual_evidence | Longitudinal Crack | 76 | 41 | 26 | 0.634 | 0.342 | 0.444 |
| rmr_dual_evidence | Potholes | 27 | 47 | 15 | 0.319 | 0.556 | 0.405 |
| rmr_dual_evidence | Transverse Crack | 32 | 29 | 8 | 0.276 | 0.250 | 0.262 |
| nafnet | Alligator Crack | 32 | 17 | 12 | 0.706 | 0.375 | 0.490 |
| nafnet | Longitudinal Crack | 76 | 19 | 13 | 0.684 | 0.171 | 0.274 |
| nafnet | Potholes | 27 | 64 | 12 | 0.188 | 0.444 | 0.264 |
| nafnet | Transverse Crack | 32 | 10 | 2 | 0.200 | 0.062 | 0.095 |
| dfpir | Alligator Crack | 32 | 16 | 11 | 0.688 | 0.344 | 0.458 |
| dfpir | Longitudinal Crack | 76 | 17 | 13 | 0.765 | 0.171 | 0.280 |
| dfpir | Potholes | 27 | 73 | 12 | 0.164 | 0.444 | 0.240 |
| dfpir | Transverse Crack | 32 | 6 | 4 | 0.667 | 0.125 | 0.211 |
| demoe_auto | Alligator Crack | 32 | 15 | 10 | 0.667 | 0.312 | 0.426 |
| demoe_auto | Longitudinal Crack | 76 | 24 | 16 | 0.667 | 0.211 | 0.320 |
| demoe_auto | Potholes | 27 | 51 | 15 | 0.294 | 0.556 | 0.385 |
| demoe_auto | Transverse Crack | 32 | 9 | 4 | 0.444 | 0.125 | 0.195 |
| demoe_scenario | Alligator Crack | 32 | 16 | 11 | 0.688 | 0.344 | 0.458 |
| demoe_scenario | Longitudinal Crack | 76 | 24 | 15 | 0.625 | 0.197 | 0.300 |
| demoe_scenario | Potholes | 27 | 50 | 13 | 0.260 | 0.481 | 0.338 |
| demoe_scenario | Transverse Crack | 32 | 9 | 3 | 0.333 | 0.094 | 0.146 |
| instructir_generic | Alligator Crack | 32 | 19 | 13 | 0.684 | 0.406 | 0.510 |
| instructir_generic | Longitudinal Crack | 76 | 22 | 15 | 0.682 | 0.197 | 0.306 |
| instructir_generic | Potholes | 27 | 43 | 13 | 0.302 | 0.481 | 0.371 |
| instructir_generic | Transverse Crack | 32 | 12 | 3 | 0.250 | 0.094 | 0.136 |
| instructir_metadata | Alligator Crack | 32 | 15 | 9 | 0.600 | 0.281 | 0.383 |
| instructir_metadata | Longitudinal Crack | 76 | 21 | 15 | 0.714 | 0.197 | 0.309 |
| instructir_metadata | Potholes | 27 | 61 | 12 | 0.197 | 0.444 | 0.273 |
| instructir_metadata | Transverse Crack | 32 | 8 | 4 | 0.500 | 0.125 | 0.200 |
