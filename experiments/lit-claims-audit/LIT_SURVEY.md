# LIT_SURVEY — feedback imprints on the low-z Lyα forest / diffuse IGM, and methods to detect them

**Track:** lit-claims-audit (branch `exp/lit-claims-audit`) · **Compiled:** 2026-10-19 · **Scope:** 2012–2026, weighted to z ≲ 1 (esp. z ≈ 0.1–0.5)
**Method:** every entry checked against its arXiv abstract page (arxiv.org/abs or the arXiv API) and, for the journal reference, against the Crossref record of its DOI. Body numbers come from the arXiv HTML / ar5iv full text, pulled by WebFetch.
**Caveat on numbers:** WebFetch hands back paraphrases from a summarizing model, not verbatim text. Numbers marked **[abs]** come from the abstract. Numbers marked **[body]** come from the full-text HTML and should be re-read against the PDF before they go into a paper. Two OCR/rendering artefacts flagged by the extraction are noted where they occur.

**Our data, as it bears on testability** (from `experiments/eigen-voigt-embedding/LEDGER.md` §2, §3 [D-08]/[D-09], §4):
- Sherwood z = 0.3, `_inf` files: 4 classes × 16384 sightlines × 2048 px, Δv = 2.6365 km/s, box 60 cMpc/h.
- Noiseless: `flux == roll(exp(-tau), 1)`. No LSF is applied to the flux on disk.
- Same skewers in every class. No temperature, density or halo fields are on disk.
- Voigt catalogs (v, b, log N) exist for the fiducial and τ-rescaled variants. τ-rescale factors to the pooled ⟨F⟩ = 0.977875 are A = [1.0297, 0.8196, 0.8579, 1.4914] for classes 1–4.
- Observed so far: median b is 31.1 km/s for class 4 vs ~24.8 for class 1; pair 2–3 reaches BA 0.575 on Voigt features; 4-class BA ≤ 0.48.

**Testability key:**
- **Y** = the claim can be tested on the on-disk data as it stands.
- **P** = partially testable. Only a direction or a difference can be checked, or a z, forward-model or suite mismatch has to be hedged.
- **N** = not testable here: it needs observations, noise/LSF realism, temperature/density fields, metals, UVB variants, halo catalogs, or a different redshift.

---

## 0. Citation corrections found in this pass (new) + carried forward

New:
1. **Nasir+2017 b-medians were mis-attributed in `Q_C3_LIT_ANCHOR.md`.** At z = 0.1 the medians are:
   - 33.5 km/s for **80-512 (QUICKLYA, no feedback)**, not for "stellar-only";
   - 35.3 km/s for 80-512-ps13+agn;
   - 30.8 km/s for 80-1024 (QUICKLYA);
   - 36.2 km/s for COS.

   So the +1.8 km/s is AGN+winds vs no-feedback. The paper gives no median for ps13 (winds only) and says winds make "very little difference" against QUICKLYA. All of these medians are measured on COS-forward-modelled mocks (COS G130M LSF, S/N = 30 per resolution element). [body]
2. **The CAMELS CDDF/b paper is Tillman+2023 (AJ 166, 228; issued 2023-11-07), not "Tillman+2024 AJ".** Crossref: DOI 10.3847/1538-3881/ad02f5. Below it is called **Tillman+2023b**; the ApJL 945, L17 paper is Tillman+2023a.
3. **The P1D paper arXiv:2410.05383 was published as ApJ 980, 72 (2025)** (DOI 10.3847/1538-4357/ada5f7). It is called **Tillman+2025a** below. The HWO paper (JATIS 11, 042219, 2025) is Tillman+2025b.
4. **Pirecki+2025 (arXiv:2509.18260) is now ApJ 1002, 63 (2026).** It has a separate companion note: Pirecki, Tillman & Burkhart, RNAAS 8, 290 (2024), "The Impact of AGN Jet Speed on the Lyα Forest Flux Power Spectrum" (Crossref metadata only; abstract not read).
5. **Sinigaglia+2026 (arXiv:2603.13011) has been accepted.** It is Phys. Rev. D (2026), DOI 10.1103/3d41-hq97; v3 is dated 2026-10-06.
6. **Maitra+2024 exists but is not a feedback paper.** It is A&A 690, A154, an IMNN on Sherwood-Relics for T0, γ, Γ_HI, σ8 and n_s at z = 2–4. The "Maitra+2026 CAMELS-SBI" citation is still the Sinigaglia+2026 mis-citation (carried forward).
7. **The Bolton+2022 arXiv ID is 2111.09600** (MNRAS 513, 864). `Q_C3` had it as "(verify)".
8. **Khaire & Srianand 2015 is MNRAS 451, L30.** Recorded here because the venue is easy to get wrong.
9. **Viel+2017 and Nasir+2017 disagree on what stellar winds do to b:**
   - V17: stellar feedback shifts b by about +3 km/s against QUICKLYA, and Sherwood AGN adds about +2 km/s more.
   - N17: ps13 winds make "very little difference" to b; adding AGN gives +1.8 km/s.
   - N17 footnote 2: V17's "peak" shift was really a shift in the median.
10. **Provenance observation (not a confirmation).** The Bolton+2022 production runs match our on-disk geometry: 60 h⁻¹ cMpc box, 2×768³ particles, 16384 sightlines × 2048 px. This bears on the HARDENING_OUTCOME flag that "60 cMpc/h is non-canonical for Bolton+2017". However:
    - the published Bolton+2022 run list has H02 (Quick-Lyα, no feedback), AGN (winds + AGN), StrongAGN, Blazar and the H00–H10 grid, but **no winds-only run**;
    - it reports simulation results only at **z = 0.1**.

    Our 4-class z = 0.3 set is therefore not identified with any published run table.

Carried forward from `Q_C3_LIT_ANCHOR.md` / `Q_S3_LIT_ANCHOR.md` (not repeated): Hummels 2020 does not exist; the intended "Sorini+2020" is Christiansen+2020; "Wadekar 2023 LyαNNA" is Nayak+2024; the AJ CAMELS paper and the P1D paper are two distinct Tillman papers.

---

## 1. Verified papers

| # | Author year | arXiv | Journal | Sim / data | z | Relevance |
|---|---|---|---|---|---|---|
| 1 | Nasir+2017 | 1706.04790 | MNRAS 471, 1056 | Sherwood P-Gadget-3, 80 h⁻¹ cMpc 2×512³: QUICKLYA / ps13 winds / ps13+agn; COS forward model | 0.1, 1.0, 1.6 | Same code family. Feedback effect on CDDF and b at low z |
| 2 | Bolton+2017 | 1605.03462 | MNRAS 464, 897 | Sherwood suite overview, 10–160 h⁻¹ cMpc, ≤17.2×10⁹ particles, outflow variants | 2–5 | Suite reference. Not low z |
| 3 | Bolton+2022 | 2111.09600 | MNRAS 513, 864 | Sherwood-project P-Gadget-3, 60 h⁻¹ cMpc 2×768³: H02 / AGN / StrongAGN / Blazar / H00–H10; COS (44 AGN) | 0–0.2 | b too narrow vs COS; AGN does not fix it; heating/turbulence limits |
| 4 | Viel+2017 | 1610.02046 | MNRAS 467, L86 | Sherwood 80 h⁻¹ cMpc (winds+AGN), HM01/HM12 (hot) runs, Illustris; COS | 0.1 | The b-distribution as a feedback diagnostic; Γ_HI ×1.5–3 HM12 |
| 5 | Viel+2013 | 1207.6567 | MNRAS 429, 1734 | OWLS (REF, AGN, wind variants) | 2.25–4 | Feedback changes flux PDF and P_F(k); effect falls with increasing z |
| 6 | Chabanier+2020 | 2002.02822 | MNRAS 495, 1825 | Horizon-AGN / -noAGN (RAMSES), 8 runs, 100 h⁻¹ Mpc | 2.0–4.25 | AGN suppresses P1D at all scales; grows toward low z |
| 7 | Tillman+2023a | 2210.02467 | ApJL 945, L17 | Simba 50 h⁻¹ Mpc (jet / no-jet), TNG (kinetic / no-kinetic); COS D16 | 0.1 | Jets set the CDD; χ²_R collapses under UVB rescaling |
| 8 | Tillman+2023b | 2307.06360 | AJ 166, 228 | CAMELS TNG + Simba, 25 h⁻¹ Mpc 256³, 1P / CV / LH sets | ≲2, focus 0.1 | AGN jets broaden b; TNG AGN has no effect |
| 9 | Tillman+2025a | 2410.05383 | ApJ 980, 72 | Simba 50 h⁻¹ Mpc 2×512³: SW-only / +Rad / +Rad+Jet / +X | 0.1–>2 | AGN on P1D; largest at z < 1 |
| 10 | Pirecki+2026 | 2509.18260 | ApJ 1002, 63 | CAMELS-Simba, 5 AGN-parameter sweeps | 0.1–2.0 | ΔP1D per AGN knob at low z |
| 11 | Tillman+2025b | 2507.13442 | JATIS 11, 042219 | Forecast (HWO UV spectrograph) | 0.4–1.8 | P1D precision needed to separate AGN models |
| 12 | Christiansen+2020 | 1911.01343 | MNRAS 499, 2617 | Simba jet / no-jet, MUFASA | 0–3 | Jets move the flux decrement D_A; photon-underproduction crisis (PUC) |
| 13 | Burkhart+2022 | 2204.09712 | ApJL 933, L46 | Illustris vs TNG100 (shared ICs and UVB) | 0.1 | CDD 2–3× difference is UVB-degenerate; b nearly identical |
| 14 | Gurvich+2017 | 1608.03293 | ApJ 835, 175 | Illustris (+ SPH comparison) | 0.1 | AGN heating mimics extra photons in the CDD |
| 15 | Khaire+2024a | 2306.05466 | MNRAS 527, 4545 | Illustris vs TNG, COS forward model, VPfit | 0.1 | Degeneracy between f_cool and Γ_HI; statistics insensitive after dN/dz matching |
| 16 | Khaire+2024b | 2311.08470 | MNRAS 534, 465 | Illustris vs TNG around LRGs | 0.1 | dN/dz(r⊥) separates feedback models at >12σ |
| 17 | Danforth+2016 | 1402.2655 | ApJ 817, 111 | HST/COS, 82 AGN, 5343 HI lines | <0.75 | Observed CDDF, b, dN/dz |
| 18 | Gaikwad+2017b (VIPER) | 1610.06572 | MNRAS 467, 3172 | COS 82 QSOs; VIPER fits; SPH sims | <0.45 | Simulated b too small; needs b_turb ≈ 20 km/s; Γ_HI(z) |
| 19 | Gaikwad+2017a | 1605.02738 | MNRAS 466, 838 | COS 82 QSOs; flux PDF + P(k); GADGET-2 + CITE | ≲0.45 | Γ_HI from flux statistics (background; no claim extracted) |
| 20 | Khaire+2019 | 1808.05605 | MNRAS 486, 769 | COS 65 QSOs, P_F(k); Nyx 20 h⁻¹ cMpc | 0.03–0.41 (incl. a z̄ = 0.30 bin) | The observed low-z P_F(k) |
| 21 | Hu+2022 | 2207.07151 | MNRAS 515, 2188 | Nyx 624 runs, DELFI on b–N + dN/dz, COS mocks | 0.1 | Simulation-based inference (SBI) on b–N for T0, γ, Γ |
| 22 | Hu+2024 | 2308.14738 | MNRAS 527, 11338 | Nyx-trained SBI applied to Illustris/TNG mocks | 0.1 | b–N robust to the WHIM / feedback |
| 23 | Hu+2025 | 2311.17895 | MNRAS 536, 1 | HST/STIS 12 QSOs, 301 lines | 0.9–1.5 | T0, γ, Γ at z ~ 1 |
| 24 | Hu+2026 | 2606.05006 | (preprint, Jun 2026) | COS 82 QSOs, SBI on b–N | 0.06–0.48 (4 bins) | Hot (~28,000 K) near-isothermal low-z IGM, or v_tur ≈ 15 km/s |
| 25 | Kollmeier+2014 | 1404.2933 | ApJL 789, L32 | Simulations vs COS | low z | PUC: Γ_HI 5× HM12 |
| 26 | Shull+2015 | 1502.00637 | ApJ 811, 3 | UVB modelling | low z | PUC / escape fractions (context) |
| 27 | Khaire & Srianand 2015 | 1503.07168 | MNRAS 451, L30 | UVB modelling | low z | QSO Γ ~2× higher; PUC less severe |
| 28 | Mallik & Srianand 2024 | 2309.05717 | MNRAS 528, 3844 | Hydro sims, aligned O VI–H I absorbers | ~0.5 | Non-thermal b depends on feedback and UVB |
| 29 | Sinigaglia+2026 | 2603.13011 | PRD (2026), 10.1103/3d41-hq97 | CAMELS LH (TNG + Simba), NPE on P1D | 2.0–3.5 | Feedback parameters return to the prior |
| 30 | Maitra+2024 | 2404.04327 | A&A 690, A154 | Sherwood-Relics, IMNN on P_F(k) | 2–4 | ML on the forest (thermal / cosmological, not feedback) |
| 31 | Nasir+2024 | 2404.05794 | MNRAS 534, 1299 | Sherwood-Relics, Bayesian CNN / ResNet per pixel | 4–5 | Per-sightline inference: T0 to ~1000 K from one sightline |
| 32 | Nayak+2024 (LyαNNA) | 2311.02167 | A&A 689, A153 | Nyx, ResNet field-level | 2.2 | ML on the forest (thermal only) |
| 33 | Jalan+2024 (FLAME) | 2403.07498 | A&A 688, A126 | CNN Voigt fitter, COS-like training | low z | ML line decomposition |
| 34 | Sharma+2026 | 2607.05494 | (submitted MNRAS, Jul 2026) | CNN absorber detection, TNG-trained, VIPER labels | low z / UVES test | ML per-absorber b, N |
| 35 | Dong+2024 | 2402.13568 | MNRAS 532, 4876 | Simba, protocluster Lyα | ~2.3 | Feedback in forest tomography (out of z range; Crossref metadata only) |
| 36 | Pirecki+2024 (RNAAS) | — | RNAAS 8, 290 | CAMELS-Simba jet speed | — | Crossref metadata only; no claim extracted |

36 papers have been checked for existence. Claims below come from papers 1–34. Papers 35–36 are verified by metadata only, and no claims are taken from them.

---

## 2. Claims ledger

Format for each claim:
- **Statistic** and the **regime** it was measured in (z, simulation, forward model).
- **Test**: testability (Y / P / N) and what would check it on our data.

**Sherwood / Sherwood-project (closest code family)**

- **C-01 Nasir+2017.** Stellar or AGN feedback, "as currently implemented", has only a small effect on the CDDF and the b distribution. [abs]
  - Statistic: CDDF and b distribution. Regime: z ≤ 1.6, Sherwood 80-512 runs.
  - **Test: Y.** Compare class 1/2/3 CDDFs and b distributions at z = 0.3. Class 4 is not in their run set.
- **C-02 Nasir+2017.** Median b at z = 0.1: 80-512 QUICKLYA 33.5 km/s; ps13+agn 35.3 km/s, so Δ ≈ +1.8 km/s (≈5%) from winds + AGN vs no feedback; 80-1024 30.8 km/s; COS 36.2 km/s. [body]
  - Statistic: median b. Regime: z = 0.1, COS LSF + S/N 30 mocks, VPFIT.
  - **Test: P.** Δmedian b(class 3 − class 1) at z = 0.3 can be measured. The absolute values need the COS forward model and z = 0.1.
- **C-03 Nasir+2017.** ps13 winds make "very little difference" to b relative to QUICKLYA. [body]
  - Statistic: b distribution. Regime: z = 0.1.
  - **Test: Y.** Class 2 vs class 1 b distributions (KS / Δmedian) on the catalogs.
- **C-04 Nasir+2017.** ps13 winds produce 2–5× more absorbers than QUICKLYA at N_HI ≳ 10^14 cm⁻². [body]
  - Statistic: CDDF ratio. Regime: z = 0.1–1.6.
  - **Test: Y.** Ratio of class 2 to class 1 CDDF at log N > 14 (fiducial and τ-rescaled).
- **C-05 Nasir+2017.** AGN gives slightly fewer N_HI > 10^14 systems than winds-only, and "only impacts the Lyα absorbers at very low redshift, z = 0.1, by producing some broader lines". [abs / body]
  - Statistic: CDDF and b. Regime: z = 0.1 vs 1.0 / 1.6.
  - **Test: P.** Check whether the class 3 vs class 2 high-N deficit and broadening are present at z = 0.3. One redshift only, so this is not an evolution test.
- **C-06 Nasir+2017.** AGN increases the number of broad lines at b > 50 km/s. [body]
  - Statistic: f(b > 50). Regime: z = 0.1.
  - **Test: Y.** f(b > 50 km/s) for class 3 vs class 2 (and class 4).
- **C-07 Nasir+2017.** At z = 0.1, going from ps13 to ps13+agn raises the WHIM mass fraction from 24.3% to 36.9% and lowers the condensed fraction from 9.1% to 3.3% and stars from 7.7% to 3.3%. [body]
  - Statistic: phase fractions.
  - **Test: N.** No temperature or density fields on disk.
- **C-08 Nasir+2017.** Under OWLS-like S/N and resolution, ps13+agn gives a median b of 27.7 km/s, against Tepper-García+2012's 29.4 km/s. The median moves by ~8 km/s with the forward model alone. [body]
  - **Test: P.** Only the forward-model sensitivity can be shown, by degrading our noiseless spectra.
- **C-09 Viel+2017.** COS b medians at z ≈ 0.1: data 36.2 km/s; Sherwood (winds + AGN, 80 h⁻¹ cMpc) 33.6; HM01 28.0; Illustris 28.3; HM01hot 34.5. HM01 and Illustris underpredict b = 25–45 km/s lines by ~2×. Stellar feedback shifts b by ~+3 km/s and AGN by ~+2 km/s more. [abs / body]
  - Regime: COS LSF + S/N 30 mocks.
  - **Test: P.** Δb(class 2 − class 1) and Δb(class 3 − class 2) at z = 0.3. This decides between V17 and N17 (C-03).
- **C-10 Viel+2017.** The CDDF prefers Γ_HI = 1.5–3× the Haardt & Madau 2012 rate. [abs]
  - **Test: N.** One UVB per class; τ-rescaling only mimics a change of Γ.
- **C-11 Viel+2017.** Strong AGN (Illustris) heats gas to a temperature-PDF peak at 10^6.5 K, too hot to absorb in Lyα. Sherwood has a secondary 10^5 K peak. The data need a median T ≈ 10^5 K at δ = 4–40. [body]
  - **Test: P.** Indirect: at matched ⟨F⟩ and line density, does class 4 lose absorbers (dN/dz) relative to class 3? No temperature field on disk.
- **C-12 Bolton+2022.** All models (H02, AGN, StrongAGN, Blazar) peak at b = 22.5 ± 2.5 km/s, against the COS peak of 27.5 ± 2.5. AGN and H02 exceed the data at the peak by 4.6σ and 5.8σ. AGN feedback "does not resolve" the narrow-line excess. [abs / body]
  - Regime: z = 0.1; 13.3 ≤ log N ≤ 14.5 and 20 ≤ b ≤ 90 km/s; COS LSF + S/N ~17.7 per pixel.
  - **Test: P.** We can check the class 1/3/4 peak bins at z = 0.3 under the same cuts. The σ values are relative to observations.
- **C-13 Bolton+2022.** At z = 0.1, T0 = 4174 K (H02), 4216 K (AGN) and 4274 K (StrongAGN), with γ − 1 = 0.58–0.59 in all three. AGN leaves the diffuse-IGM T–ρ relation essentially unchanged. [body]
  - **Test: P.** Use the b–N lower envelope (10th-percentile b per log N bin) as a T0 proxy. If the class 1/3/4 envelopes coincide while the medians differ, the class 4 broadening sits outside the T0 relation.
- **C-14 Bolton+2022.** CDDF-matched log Γ_HI is −13.22 for both H02 and AGN, but −13.50 for StrongAGN (×0.52). Relative to HM12 the ratios are 1.73, 1.69 and 0.90. [body]
  - Statistic: Γ needed to match the CDDF over 13.3 ≤ log N ≤ 14.5. Regime: z = 0.1.
  - **Test: Y, quantitative.** Our mean-flux τ-scalings give A4/A3 = 1.74 (z = 0.3, matched on ⟨F⟩, not on the CDDF), and A3/A1 = 0.83, whereas Bolton has AGN = H02. A CDDF-matched rescaling over the same N range would make the comparison like for like.
- **C-15 Bolton+2022.** WHIM baryon fraction: 21.2% (H02), 37.9% (AGN), 68.4% (StrongAGN). The fraction of fitted absorbers that sit in the WHIM barely moves: 10.2%, 15.2%, 16.8%. [body]
  - **Test: N.** Needs gas phases.
- **C-16 Bolton+2022.** AGN barely changes the CDDF at N ≳ 10^13.3 relative to H02. StrongAGN changes both CDDF shape and amplitude, but stays within ~1σ of the COS errors. AGN and blazar heating flatten the CDDF near 10^14. [body]
  - **Test: Y.** CDDF of class 3 vs class 1, and class 4 vs class 3, both fiducial and τ-rescaled.
- **C-17 Bolton+2022.** Either an extra heat input u ≲ 6.9 eV m_p⁻¹ (z ≲ 2.5, N ≈ 10^13.5), or v_turb ≲ 8.5 km/s × (N/10^13.5)^0.21. The assumed b_turb ≈ 0.73 b_noturb gives ≈ 12.1 km/s at 10^13.5. [abs / body]
  - **Test: N.** Requires observations.
- **C-18 Bolton+2022.** The b distribution is converged within the COS 1σ errors at L = 60 h⁻¹ cMpc and M_gas = 6.38×10⁶ h⁻¹ M☉, and b decreases as mass resolution increases. The CDDF needs 10–30% corrections at N ≥ 10^14 (per-bin factors 1.00–1.42). [body]
  - **Test: N.** Single resolution on disk. This bears on how far absolute b and CDDF can be trusted.
- **C-19 Bolton+2022.** The required Γ is 0.4–0.6× Puchwein+2019 at z = 0.1. [abs / body]
  - **Test: N.**

**SIMBA / IllustrisTNG / CAMELS**

- **C-20 Tillman+2023a.** At z = 0.1 the Simba CDD χ²_R against COS is 4.5 (jets) vs 86 (no jets) as run. Fitting the UVB separately for each run (1.2× vs 2.7× HM12) brings them to 3.9 vs 8.6: the ratio falls from ~19 to ~2.2 and stays at 2.2 in every N range tested. Removing TNG's kinetic mode has a negligible effect. [abs / body]
  - Statistic: column densities from SPH projection, not Voigt fits.
  - **Test: P.** Analogue: the ratio of between-class CDDF distances, fiducial vs τ-rescaled.
- **C-21 Tillman+2023b.** In CAMELS-Simba a higher jet-speed parameter broadens the b distribution, more strongly than the momentum-flux knob. Weaker Simba stellar feedback shifts b up. TNG AGN variations have no effect. Simulated b ~20 km/s vs ~30 observed. The Simba and TNG CDDs diverge below z ≈ 0.7. [abs / body]
  - Regime: z = 0.1, noiseless 25 h⁻¹ Mpc boxes.
  - **Test: P.** Only the direction "AGN broadens b" is testable (class 4 vs 3, class 3 vs 2). The stellar-feedback strength cannot be varied in our set.
- **C-22 Tillman+2023b.** The UVB multiplier needed to fit the CDD at z = 0.1 spans roughly 0.3–3 across the CAMELS-Simba feedback variations (Table 2). This shows the CDD feedback signal is UVB-degenerate. [body; table labels paraphrased]
  - **Test: P.** Via the τ-rescale analogue only.
- **C-23 Tillman+2025a.**
  - Simba jets (+Rad+Jet) drop P1D by ~60% at all k (60 ± 9% at z = 0.1), relative to stellar-winds-only.
  - Radiative mode: <10%, only at k ≲ 10⁻² s/km. X-ray heating: a further ~10%.
  - The effects become observable below z ≈ 1. After rescaling τ_eff, the remaining differences are <20% at z < 1.
  - At z > 2: 2% at k < 5×10⁻² s/km and 8% above. [abs / body]
  - **Test: P.** Shape and size of P_F(k) ratios class 3/2 and 4/2 at z = 0.3, fiducial vs τ-rescaled. The Sherwood AGN model (thermal radio bubbles) is not Simba's jets, so magnitudes do not transfer.
- **C-24 Tillman+2025a.** Jets raise T0 and the mean b. With X-ray heating T0 returns close to the SW-only value, but b stays elevated. [body]
  - **Test: P.** Combined with C-13: is our class 4 broadening found at a fixed b–N lower envelope?
- **C-25 Pirecki+2026.** At z = 0.1 without mean-flux correction:
  - lower jet speed → +50–100% in P1D;
  - lower momentum flux → up to +80%;
  - a higher jet mass threshold → +50–100%;
  - radiative efficiency and jet T_vir velocity cross 10% only at z ≤ 0.4.

  After mean-flux correction (z = 0.1), the P1D ratios span max 1.03–1.22 and min 0.62–0.93, with crossover scale k0 ≈ 1.4–3.9×10⁻² s/km. Only three knobs stay above 10%. [body]
  - **Test: P.** Check whether the class-ratio crossover k0 in our τ-rescaled P_F(k) falls at 1–4×10⁻² s/km.
- **C-26 Khaire+2024a.**
  - Γ_HI needed to match dN/dz = 205.9 at z = 0.1: 3.1×10⁻¹⁴ s⁻¹ (Illustris) vs 7.73×10⁻¹⁴ (TNG), a ratio of 2.49.
  - Cool baryon fraction: 23% vs 39%. The forest constrains f_cool²/Γ_HI.
  - After matching dN/dz, b, N_HI, line-density and P_F(k) statistics are "largely insensitive" to feedback.
  - P_F(k) agrees within ~5% at k < 0.02 s/km, differs by ~4–5% above that, and exceeds the errors only at k > 0.04.
  - T0 = 4231 vs 4010 K. [abs / body]
  - **Test: Y, decisive.** Rescale each class to a common dN/dz (line density above a log N threshold) instead of a common ⟨F⟩. Then check whether the class 2–3 and class 4 b-distribution differences survive.
- **C-27 Burkhart+2022.**
  - At z = 0.1 Illustris has 2–3× fewer absorbers than TNG at N < 10^15.5; scaling the Illustris UVB by ×0.4 reproduces the TNG CDD.
  - The flux PDF and P_F(k) change shape in ways a UVB scaling does not reproduce.
  - b distributions both peak at ~20 km/s and are identical at 10^13–10^14.
  - TNG matches the COS P_F(k) better over 8.8×10⁻⁴ < k < 0.07 s/km. [abs / body]
  - **Test: Y.** After τ-rescaling, do between-class CDDF differences vanish while flux-PDF and P_F(k)-shape differences remain? H3 already partially tests the P_F(k) side.
- **C-28 Gurvich+2017.** At z = 0.1, Illustris AGN feedback alters the gas distribution to mimic extra ionizing photons. The model matches the CDD at 10^12.5–10^13.5 but gives too few absorbers at 10^14–10^16. Supernova feedback, Voigt fitting and resolution have little effect. [abs]
  - **Test: P.** Direction of the low-N CDDF shift and the high-N deficit for class 4 vs class 1.
- **C-29 Christiansen+2020.** At z = 0 without jets, Γ_HI must be 6× HM12 to match D_A; with jets, ~2.5×. Jets lower the diffuse-IGM baryon fraction from 39% to 16% and raise the WHIM from 30% to 70%. The PUC appears at z ≲ 1, where the jet and no-jet runs diverge. [abs]
  - **Test: P.** Use the per-class D_A at z = 0.3 (A_c, ⟨F⟩) as the analogue. Our "AGN" is not Simba's jets.
- **C-30 Khaire+2024b.** dN/dz(r⊥) around LRGs is 1.5–2× higher in TNG than in Illustris out to ~10 pMpc at z = 0.1. Archival data could separate the two at >12σ. [abs]
  - **Test: N.** No halo / LRG catalog on disk.

**High-z P_F(k) feedback anchors (z mismatch)**

- **C-31 Viel+2013.** In OWLS at z = 2.25, AGN changes the flux PDF by ~5% and suppresses P_F by ~10% at k ≈ 0.1 h/Mpc and ~5% at 0.6 h/Mpc. SN feedback moves the power the opposite way. The effects shrink with increasing z. [abs / body]
  - **Test: N** (z). Only the sign pattern (winds vs AGN act in opposite directions on P_F) is P.
- **C-32 Chabanier+2020.** In Horizon-AGN, AGN suppresses P1D at all scales. Large-scale suppression grows from <1% at z = 4.25 to 8% at z = 2.0, and is smaller on small scales. The hot IGM fraction is 18% vs 12%. Ignoring AGN biases σ8 by ~2% and n_s by ~1%. [abs / body]
  - **Test: N** (z; no low-z output). P only for the sign "AGN lowers power".

**Observations (low z)**

- **C-33 Danforth+2016.**
  - CDDF: β = 1.65 ± 0.02, C14 = 25 ± 1; β(z) = 1.73 − 0.26z for z < 0.47, so β(0.3) ≈ 1.65.
  - Median b: 29.6 ± 0.5 km/s at log N ≤ 13.5 and 33.8 ± 0.5 at > 13.5.
  - Broad Lyα fraction (b > 40): ~26–28% (upper limits).
  - COS resolution ~17 km/s, S/N ~20. [abs / body]
  - **Test: P.** CDDF slope at z = 0.3 from our catalogs vs β(0.3). The b comparison needs the LSF and noise. The LEDGER already records a 2× line-density shortfall vs Davé+1999.
- **C-34 Gaikwad+2017b (VIPER).**
  - Observed median b over 0.075 ≤ z ≤ 0.45: 32.9 ± 20.8 km/s (VIPER) and 33.9 ± 18.3 (D16). Simulated b is "significantly smaller".
  - Adding Gaussian b_turb (mean 20, σ 10 km/s), or multiplying T by 3, reproduces the b distribution and the b–N lower envelope at log N ≥ 13.2 (χ²/dof 2.08 vs 4.17).
  - Γ12 = 0.137 ± 0.015 at z̄ = 0.3; Γ_HI(z) = 3.9×10⁻¹⁴ (1+z)^4.98 s⁻¹. [abs / body]
  - **Test: P.** Gap between our class medians and 32.9 km/s; absolute b needs the COS forward model.
- **C-35 Khaire+2019.**
  - COS P_F(k) measured in 5 bins, including 0.26–0.36 (z̄ = 0.30), over ~5×10⁻⁴–0.15 s/km.
  - Clear small-scale cutoff at k > 0.03 s/km in all bins.
  - Γ_HI(z = 0.30) = 1.479 (+0.44/−0.52)×10⁻¹³ s⁻¹ from a Nyx 20 h⁻¹ cMpc box. [abs / body]
  - **Test: P.** Location of the cutoff in our per-class P_F(k) at z = 0.3, and whether the class 4 broadening moves it to lower k. Matching the absolute amplitude needs the LSF and noise.
- **C-36 Hu+2026.** From COS (82 QSOs, z = 0.06–0.48, 4 bins), at z = 0.1:
  - log T0 = 4.45 (+0.08/−0.12), i.e. T0 ≈ 28,000 K, and γ = 1.06, about 7σ above the canonical log T0 ≈ 3.60.
  - log Γ = −13.70, about 4σ below the synthesis models (≈ −13.3).
  - Alternatively, v_tur ≈ 15 km/s added to a standard thermal state reproduces the widths. STIS comparisons argue against an instrumental cause. [abs]
  - **Test: N** (observational). P only for whether any of our classes reaches an effective T0 of this size, judged from its b–N envelope.
- **C-37 Hu+2025.** STIS, 0.9 < z < 1.5: [log T0, γ] = [4.13, 1.34], [3.79, 1.70] and [4.12, 1.34] at z = 1.4, 1.2 and 1.0; log Γ = −11.79, −11.98 and −12.32. The T0 errors (~5000 K) prevent concluding whether the IGM cools. [abs]
  - **Test: N.**

**Methods / ML / SBI**

- **C-38 Hu+2022.** DELFI on the b–N distribution plus dN/dz, trained on 624 Nyx runs at z = 0.1. On 34-sightline COS mocks it reaches σ(log T0) ≈ 0.08 dex, σ(γ) ≈ 0.06 and σ(log Γ) ≈ 0.07 dex. [abs]
  - **Test: P.** The method can run on our catalogs, but the training set has no feedback.
- **C-39 Hu+2024.** A Nyx-trained b–N SBI recovers [T0, γ] for Illustris and TNG mocks with |Δlog T0| ≲ 0.05 dex and |Δγ| ≲ 0.1, even though volume-weighted WHIM fractions are 38%, 10% and 4% (Illustris / TNG / Nyx). The b–N distribution is "not sensitive to the WHIM". [abs]
  - **Test: P.** If this holds for Sherwood, the class 4 vs 1 b–N lower-envelope offsets should be small (≲0.05 dex in inferred T0) even though class 4's median b is shifted. This is the same test as C-13.
- **C-40 Sinigaglia+2026.** Neural posterior estimation (NPE) on P1D with CAMELS LH (TNG + Simba) at z = 2–3.5. The four feedback parameters revert to the prior, which the authors attribute to the limited volume. Ωm and σ8 are within 10% in ≥75% and ≥90% of cases. Cross-model training biases σ8 by ~10%. [abs]
  - **Test: N** (z, continuous parameters). An analogue posterior width on stacked P_F(k) is only P.
- **C-41 Maitra+2024.** An IMNN on Sherwood-Relics P_F(k) gives better T0 and γ than MCMC at z = 2–3, at FWHM = 6 km/s. [abs]
  - **Test: N** (no feedback target).
- **C-42 Nasir+2024.** A Bayesian CNN/ResNet reconstructs per-pixel density and temperature. From a single 20 h⁻¹ cMpc sightline it estimates T0 to ~1000 K (1σ), with ~100× less pathlength than standard methods. Regime: Sherwood-Relics, 4 ≤ z ≤ 5. [abs]
  - **Test: N** (z, and no temperature labels on disk). It is the closest published per-sightline inference precedent.
- **C-43 Nayak+2024.** A ResNet field-level inference of T0 and γ at z = 2.2 (Nyx). [abs]
  - **Test: N.**
- **C-44 Jalan+2024 (FLAME).** A CNN Voigt fitter correctly identifies >98% of single-component and ~90% of double-component lines. For 90% of single components, b is within ±8 km/s and log N within ±0.3. Classification accuracy drops ~10% on real COS data. [abs]
  - **Test: P.** Only as an independent cross-check of our VoigtFit catalogs. Not a feedback claim.
- **C-45 Sharma+2026.** A CNN absorber finder trained on TNG with VIPER labels reaches F1 ≈ 0.8, mean absolute errors of 0.18 dex in log N and 0.10 dex in log b, and a b–N envelope RMS of 0.36 km/s on simulations. On UVES data F1 falls to ≈ 0.5 and the envelope RMS rises to 2.96 km/s. [abs]
  - **Test: P** (tooling only).
- **C-46 Mallik & Srianand 2024.** For aligned O VI–H I absorbers at z ~ 0.5, the inferred thermal temperature tracks the τ-weighted kinetic temperature with large scatter. The non-thermal b depends on both feedback and the UVB, so it is not a clean turbulence indicator. [abs]
  - **Test: N.** Needs metals (O VI).
- **C-47 Tillman+2025b.** Distinguishing AGN models needs ~5–10% P1D precision at z ≤ 1.8, which means ~830 QSO spectra at S/N ≥ 5–25 with R = 40,000. [abs]
  - **Test: N** (forecast).

**UVB / PUC context**

- **C-48 Kollmeier+2014.** Low-z simulations need Γ_HI ~5× HM12. The arXiv and published abstracts disagree on the sign of the flux-decrement mismatch. [abs]
  - **Test: N.**
- **C-49 Khaire & Srianand 2015; Shull+2015.** Revised QSO emissivity raises the QSO contribution to Γ by ~2×, so the PUC is "not as severe". [abs]
  - **Test: N.**

**Tally:** 49 claims. **Y = 8** (C-01, 03, 04, 06, 14, 16, 26, 27). **P = 22** (C-02, 05, 08, 09, 11, 12, 13, 20–25, 28, 29, 33–35, 38, 39, 44, 45). **N = 19**. Testable (Y + P) = 30.

---

## 3. Open problems and tensions named by the literature itself

1. **Low-z line-width (b) tension.** Simulated Lyα lines at z ≈ 0.1 are too narrow compared with COS, across codes and feedback models:
   - simulated peak 22.5 vs observed 27.5 km/s (Bolton+2022);
   - median b 28–34 vs 36.2 km/s (Viel+2017; Nasir+2017);
   - CAMELS ~20 vs ~30 km/s (Tillman+2023b);
   - Illustris/TNG peak ~20 km/s (Burkhart+2022; Khaire+2024a);
   - simulated b "significantly smaller" than observed (Gaikwad+2017b).

   Feedback as implemented does not close the gap (Nasir+2017; Bolton+2022, including StrongAGN and Blazar). Tillman+2025a and +2025b say no simulation matches the observed b distribution at z < 1.
2. **Extra heating vs unresolved turbulence.**
   - Heating options: Δ = 4–40 gas at ~10^5 K (Nasir+2017; Viel+2017); u ≲ 6.9 eV m_p⁻¹; a 3× temperature boost (Gaikwad+2017b).
   - Turbulence options: v_turb ≲ 8.5 km/s (Bolton+2022); b_turb ~20 ± 10 km/s (Gaikwad+2017b); v_tur ≈ 15 km/s (Hu+2026).
   - Hu+2026 alternatively infers a hot, near-isothermal IGM, T0 ≈ 28,000 K and γ ≈ 1.06, about 7σ above the canonical value.
   - Enhanced photoheating requires an unphysically hard UVB (Bolton+2022).
   - Non-thermal broadening is itself feedback- and UVB-dependent (Mallik & Srianand 2024).
3. **UVB vs feedback degeneracy.** The CDD response to AGN can be mimicked by rescaling the UVB:
   - Burkhart+2022: Illustris UVB ×0.4 reproduces the TNG CDD;
   - Tillman+2023a: χ²_R ratio 19 → 2.2;
   - Tillman+2023b: UVB factors vary widely across feedback knobs;
   - Tillman+2025a: "highly degenerate".

   Khaire+2024a gives the mechanism: the forest constrains f_cool²/Γ_HI, and after matching dN/dz most statistics become insensitive. Proposed ways to break it: flux PDF and P_F(k) shape (Burkhart+2022); small-scale P_F(k) at k > 0.04 s/km (Khaire+2024a); independent Γ or f_cool (Khaire+2024a); dN/dz(r⊥) around LRGs (Khaire+2024b).
4. **Photon-underproduction crisis (PUC) and Γ_HI calibration.** Required Γ_HI vs HM12:
   - ~5× (Kollmeier+2014);
   - 1.5–3× (Viel+2017);
   - ~1.7× (Bolton+2022);
   - 6× → 2.5× once jets are on (Christiansen+2020).

   The crisis is eased by revised QSO emissivities (Khaire & Srianand 2015; Shull+2015) and by AGN heating (Gurvich+2017; Christiansen+2020). The observed P_F(k) is claimed to be explainable by QSOs alone (Khaire+2019). The newest b–N SBI value (log Γ = −13.70 at z = 0.1) sits ~4σ below synthesis models (Hu+2026).
5. **Inconsistent stellar-wind effect on b within the Sherwood family.** V17 reports about +3 km/s from stellar feedback; N17 reports "very little difference" (§0 item 9).
6. **Resolution and box dependence of b and the CDDF.** b decreases as mass resolution increases, and the CDDF needs 10–30% corrections at N ≥ 10^14 (Bolton+2022). Nasir+2017 gives median b 30.8 (1024³) vs 33.5 km/s (512³). Simba z = 0.1 P1D shifts ~40% with resolution before τ_eff rescaling (Tillman+2025a). The CDD depends strongly on the UVB and feedback in 25 h⁻¹ Mpc CAMELS boxes, and cosmic variance swamps feedback parameters there (Tillman+2023b; Sinigaglia+2026).
7. **Model-specificity.** Large low-z P1D effects are tied to Simba's long-range jets (Tillman+2023a, +2025a; Pirecki+2026). TNG kinetic-mode variations leave the forest unchanged (Tillman+2023a, +2023b). Khaire+2024a notes its small-scale P_F(k) differences may be specific to Illustris's extreme feedback.
8. **Redshift coverage gap.** Most feedback-on-forest numbers are at z = 0.1. Tillman+2025b names the missing data at z ~ 0.4–1.8. Nasir+2017 finds the AGN effect only at z = 0.1 among its outputs at z = 0.1, 1.0 and 1.6. Tillman+2023b finds the Simba and TNG CDDs diverge below z ≈ 0.7.
9. **No published per-sightline or line-by-line feedback discriminator was found.**
   - Existing feedback discrimination is ensemble-level: CDDF, b distributions, P_F(k), flux PDF, dN/dz(r⊥).
   - Per-sightline ML exists only for thermal parameters (Nasir+2024; Nayak+2024; Maitra+2024) and for line decomposition (Jalan+2024; Sharma+2026).
   - SBI on feedback parameters returns the prior at z = 2–3.5 (Sinigaglia+2026).

---

## 4. UNVERIFIED — do not cite

| Item | Status |
|---|---|
| "Hummels et al. 2020" (low-z CDDF/b feedback anchor) | Does not exist as described (carried from Q_C3) |
| "Sorini, Davé, Anglés-Alcázar 2020 — SIMBA AGN-vs-no-AGN low-z flux statistics" | Mis-citation; the intended paper is Christiansen+2020 (carried) |
| "Maitra+2026 CAMELS-SBI" | Mis-citation of Sinigaglia+2026. Maitra+2024 exists but is not a feedback paper (carried, refined) |
| "Wadekar 2023 LyαNNA" | Mis-citation of Nayak+2024 (carried) |
| "Tillman+2024 AJ" / "Tillman+2024 ApJ" (as year labels) | The papers exist, but the years are wrong: AJ 166, 228 is 2023, and 2410.05383 is ApJ 980, 72 (2025) |
| "Hu+2024" as an ML feedback-inference paper | No such paper. The Hu+ 2022–2026 series infers thermal state and Γ from b–N, not feedback |
| A CNN (or any ML) classifier of feedback model from low-z Lyα spectra | Not found after several targeted searches (standard + extended) |
| A per-sightline / line-by-line feedback-discrimination study | Not found (see §3 item 9) |
| "Kim, Wakker et al. 2021" COS line sample (used by Viel+2017 / Bolton+2022) | Not located by search; cite Bolton+2022 / Viel+2017 for the 44-AGN COS sample instead |
| Q_C3's "AGN median 35.3 vs stellar-only 33.5 km/s" | Incorrect attribution (§0 item 1); 33.5 is the no-feedback QUICKLYA run |
| Bolton+2022 AGN radio-mode bubble fractions "2% / 8%" | The HTML renders them as "22% / 88%", which is probably a rendering error. Confirm in the PDF before citing |
| Pirecki+2024 RNAAS 8, 290; Dong+2024 (2402.13568) | Metadata verified via Crossref only; no claims taken |
