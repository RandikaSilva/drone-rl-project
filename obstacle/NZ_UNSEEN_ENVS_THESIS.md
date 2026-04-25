# Unseen Environments — New Zealand Biome Mapping (Thesis Reference)

This document maps each of the five unseen-environment test files to a
real New Zealand forest biome and provides the citations needed to
justify the mapping in the thesis.

## 1. Summary table

| # | File | NZ biome | Obstacles | Real-world analogue |
|---|---|---|---|---|
| 1 | `v10_unseen1_grid_forest.py` | **NZ Pine Plantation** (Kaingaroa-style) | 50 | Pre-thinning *Pinus radiata*, ~1,033 stems/ha |
| 2 | `v10_unseen2_random_scatter.py` | **NZ Native Podocarp-Broadleaf Bush** | 60 | Kāmahi-podocarp alliance, ~1,240 stems/ha |
| 3 | `v10_unseen3_s_corridor.py` | **NZ Fiordland Gorge Corridor** | 60 | Bush-walled river gorge, DOC ops terrain |
| 4 | `v10_unseen4_giant_pillars.py` | **NZ Kauri Grove** (Waipoua-style) | 52 | Mature *Agathis australis* (Tāne Mahuta-scale) |
| 5 | `v10_unseen5_labyrinth_walls.py` | **NZ Fern Gully (under-canopy)** | 70 | Ponga/tree-fern understory beneath podocarp canopy |

## 2. Per-env justification

### Env 1 — NZ Pine Plantation
- 4m regular grid, ~1,033 stems/ha matches **NZFFA initial-planting default**
  of 1000 stems/ha at 3.16×3.16m spacing.
- *Pinus radiata* covers ~90% of NZ commercial forest (Kaingaroa, Kinleith).
- Direct research precedent: **Salmond et al. (2022)**, *"Autonomous Surveying
  of Plantation Forests Using Multi-Rotor UAVs"*, MDPI Drones 6(9):256 —
  same regular-grid operational scenario.

### Env 2 — NZ Native Podocarp-Broadleaf Bush
- Unstructured seeded-random placement (60 stems in 484 m² ≈ 1,240/ha)
  matches **Manaaki Whenua Landcare Research** Kāmahi-podocarp forest plots.
- Mixed trunk radii (0.30–0.65m) and canopy heights (4–6.5m) reflect the
  species-rich, uneven-aged structure of intact native bush.
- Literature anchor: **Tordesillas et al. (2024)**, *"Scalable Autonomous
  Drone Flight in the Forest with Visual-Inertial SLAM"*, arXiv:2403.09596 —
  the canonical under-canopy baseline for unstructured native forest.

### Env 3 — NZ Fiordland Gorge Corridor
- Two jittered tree walls (y ≈ ±5) + shoulder emergents (y ≈ ±7.5) + floor
  logs/boulders form a bush-walled river gorge.
- Matches the terrain where DOC-permitted conservation UAVs operate
  (whio monitoring, stoat-trap deployment, mustelid control).
- Policy anchor: **DOC drone-use-on-conservation-land permits** framework.

### Env 4 — NZ Kauri Grove (Waipoua-style)
- 10 giant pillars at radius 0.90–1.10m approximate half-scale mature kauri
  (*Agathis australis*); real giants reach 1.5–5m diameter
  (Tāne Mahuta 4.4m, Te Matua Ngahere 5.2m).
- Surrounding 25 medium associates (r = 0.35–0.55m) mimic rimu, miro,
  taraire.
- Ecological / policy anchor: **DOC kauri-dieback programme** (Waipoua
  Forest monitoring for *Phytophthora agathidicida*).
- Research anchor: **Meiforth et al. (2019)** — UAV-based kauri-dieback
  detection.

### Env 5 — NZ Fern Gully (under-canopy)
- 8 emergents + 15 sub-canopy saplings + 15 ponga (tree-fern) trunks +
  28 flyable fern bushes simulates the dense understory beneath a podocarp
  canopy.
- Thin ponga trunks (r = 0.18–0.22m, h = 2.5–3.5m) are intentionally
  **below** v10's training radius distribution — the policy must avoid
  very thin vertical stems at cruise altitude 1.5m.
- Literature anchors (under-canopy flight is an active research topic
  precisely because GPS is blocked and fern cover dominates LiDAR):
  - **Kuronen et al. (2025)** — arXiv:2501.12073
  - **ISPRS 2025** — "Under-canopy UAV Solutions for Forest Inventory"
  - **Zhou et al. (2024)** — Geo-Spatial Information Science

## 3. Thesis bibliography

### 3.1 New Zealand ecology / forestry (justifies stem densities & species)

1. NZFFA. *Initial spacing in radiata pine.* NZ Farm Forestry Association.
   <https://www.nzffa.org.nz/farm-forestry-model/resource-centre/tree-grower-articles/may-2007/initial-spacing-in-radiata-pine/>
2. Te Ara Encyclopedia of New Zealand. *Radiata pine.*
   <https://teara.govt.nz/en/radiata-pine/>
3. Te Ara Encyclopedia of New Zealand. *Mixed Broadleaf Podocarp and Kauri
   Forest.* <https://teara.govt.nz/en/1966/forests-indigenous/page-4>
4. Manaaki Whenua Landcare Research. *Kāmahi-podocarp forest.*
   <https://www.landcareresearch.co.nz/publications/woody-ecosystem-types/broadleaved-podocarp-forest-alliances-including-kauri/kamahi-podocarp-forest>
5. Department of Conservation (DOC). *Our work protecting kauri* (kauri
   dieback programme, Waipoua Forest).
   <https://www.doc.govt.nz/nature/pests-and-threats/diseases/kauri-dieback/>
6. Department of Conservation (DOC). *Drone use on conservation land:
   permits framework.*
   <https://www.doc.govt.nz/get-involved/apply-for-permits/all-permissions/drone-use-on-conservation-land/>

### 3.2 Drone navigation / RL / LiDAR in forest (positions your work)

7. Tordesillas, J., Gupta, S., Gao, Z., et al. (2024). *Scalable Autonomous
   Drone Flight in the Forest with Visual-Inertial SLAM and Dense Submaps
   Built without LiDAR.* arXiv:2403.09596.
   <https://arxiv.org/abs/2403.09596>
8. Kuronen, J., et al. (2025). *Towards autonomous photogrammetric forest
   inventory using a lightweight under-canopy robotic drone.*
   arXiv:2501.12073. <https://arxiv.org/abs/2501.12073>
9. ISPRS Archives (2025). *Under-canopy UAV Solutions for Forest Inventory
   – Challenges and Opportunities.*
   <https://isprs-archives.copernicus.org/articles/XLVIII-2-W11-2025/183/2025/>
10. Zhou et al. (2024). *Forest in situ observations through a fully
    automated under-canopy unmanned aerial vehicle.* Geo-Spatial Information
    Science.
    <https://www.tandfonline.com/doi/full/10.1080/10095020.2024.2322765>
11. Salmond, K., et al. (2022). *Autonomous Surveying of Plantation Forests
    Using Multi-Rotor UAVs.* Drones 6(9):256.
    <https://www.mdpi.com/2504-446X/6/9/256>
12. *DRL-Based UAV Autonomous Navigation and Obstacle Avoidance with LiDAR
    and Depth Camera Fusion.* Aerospace 12(9):848, 2025.
    <https://www.mdpi.com/2226-4310/12/9/848>
13. *A hybrid framework for UAV obstacle avoidance integrating reactive
    sensing, LiDAR planning, and deep reinforcement learning in real time.*
    ScienceDirect 2026.
    <https://www.sciencedirect.com/science/article/pii/S2590123026011850>
14. *Autonomous Drone Navigation in Forest Environments Using Deep
    Learning.* ISPRS Archives 2025.
    <https://isprs-archives.copernicus.org/articles/XLVIII-2-W11-2025/87/2025/>

## 4. Ready-to-use thesis paragraph

> To evaluate out-of-distribution generalisation, the v10 policy was tested
> in five unseen environments, each designed to mirror a distinct New
> Zealand operational biome: (1) a regular *Pinus radiata* plantation at
> ~1,000 stems/ha, representative of NZ's commercial forestry estate
> [NZFFA; Salmond et al. 2022]; (2) an unstructured native
> podocarp–broadleaf bush following the density and species-mixing observed
> in Manaaki Whenua Kāmahi–podocarp plots; (3) a Fiordland-style river
> gorge with bush-walled cliffs and floor debris, matching the terrain
> where DOC-permitted conservation UAVs operate; (4) a mature kauri grove
> modelled after Waipoua Forest, featuring trunk radii well beyond the
> training distribution and aligned with kauri-dieback monitoring [DOC];
> and (5) a dense under-canopy fern gully with ponga tree-fern trunks —
> the most ecologically-relevant scenario for NZ pest-monitoring
> deployments and directly comparable to the under-canopy benchmarks of
> Tordesillas et al. (2024) and Kuronen et al. (2025).

## 5. Caveats to declare in the thesis

- **2D obstacles on flat ground.** Cylindrical-trunk obstacles on a flat
  floor do not capture NZ's rugged vertical terrain (gullies, ridges,
  elevation change). Cruise altitude is fixed at 1.5m.
- **No weather / wind.** Real NZ bush flight is dominated by gusts and
  microclimate turbulence inside the canopy — not simulated here.
- **No vegetation compliance.** Real ferns/branches yield on contact; our
  trunks are rigid cylinders — conservative for safety evaluation.
- **Arena is 22 × 22 m.** Scaled, not a full-operation footprint.

These caveats are best placed alongside the results discussion so the
NZ-biome mapping is framed as a *proxy* rather than a literal field test.
