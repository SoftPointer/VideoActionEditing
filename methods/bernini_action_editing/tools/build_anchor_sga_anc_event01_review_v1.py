#!/usr/bin/env python3
"""Build a compact synchronized Event 01 mechanism review page."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
from pathlib import Path


SECTIONS = (
    (
        "Input authorities and frozen baseline",
        (
            ("Source authority", "source.mp4", "Identity, scene, frame 0 and all non-action content"),
            ("Pure-T2V action anchor", "anchor.mp4", "Action demonstration; appearance is not authority"),
            ("Frozen RV2V", "frozen.mp4", "Fails Event 01 by continuing to walk"),
        ),
    ),
    (
        "Round 1 · hard Q/K · blocks 0–15",
        (
            ("Hard QK · IID1", "AQK_IID1.mp4", "40 online anchor forwards; source scene erased"),
            ("Hard QK · ANC1", "AQK_ANC1.mp4", "ANC-only comparison"),
            ("Hard QK · AVG5", "AQK_AVG5.mp4", "Five early candidates, uniform aggregation"),
            ("Hard QK · SGA5", "AQK_SGA5.mp4", "Five early candidates, source-similarity aggregation"),
        ),
    ),
    (
        "Round 2 · sparse frame-0 temporal residual · ANC1",
        (
            ("TR-QK α.10 · mid 8–21", "TRQK_A010_MID8_21.mp4", "Top-25% temporal residual; exact phase-0 current Q/K"),
            ("TR-QK α.25 · mid 8–21", "TRQK_A025_MID8_21.mp4", "Stronger matched mid-block route"),
            ("TR-K α.25 · mid 8–21", "TRK_A025_MID8_21.mp4", "Current Q/V; anchor temporal residual only in K"),
            ("TR-QK α.10 · late 20–29", "TRQK_A010_LATE20_29.mp4", "Weak late-block route"),
        ),
    ),
    (
        "Round 3 · early-only action route + exact initial-state clamp · ANC1",
        (
            ("FlowEdit + ANC · no anchor", "FLOWEDIT_ANC1_NO_ANCHOR.mp4", "Scientific control: zero pure-T2V forwards"),
            ("TR-QK α.10 · first 3 steps", "TRQK_A010_EARLY3_MID8_21.mp4", "Anchor is queried online for every candidate in steps 0–2"),
            ("TR-QK α.10 · first 8 steps", "TRQK_A010_EARLY8_MID8_21.mp4", "Longer early action route; no late attention overwrite"),
            ("TR-K α.25 · first 3 steps", "TRK_A025_EARLY3_MID8_21.mp4", "Anchor changes K routing only; current Q/V retained"),
        ),
    ),
    (
        "Round 4 · CFG-balanced online anchor route · ANC1",
        (
            ("Balanced TR-QK α.10 · first 3 steps", "TRQK_CFG2_A010_EARLY3_MID8_21.mp4", "Same anchor Q/K conditions target negative and positive forwards"),
            ("Balanced TR-QK α.10 · first 8 steps", "TRQK_CFG2_A010_EARLY8_MID8_21.mp4", "Anchor route is no longer amplified as text-CFG residual"),
        ),
    ),
    (
        "Round 5 · outer-sampler diagnosis without latent phase clamp · ANC1",
        (
            ("FlowEdit + ANC · no anchor · no clamp", "FLOWEDIT_ANC1_NO_ANCHOR_NO_CLAMP.mp4", "Tests whether the hard temporal boundary caused corruption"),
            ("Balanced TR-QK α.10 · first 3 · no clamp", "TRQK_CFG2_A010_EARLY3_NO_CLAMP.mp4", "Online anchor under the same unclamped outer sampler"),
        ),
    ),
    (
        "Round 6 · raw conditional FlowEdit field · no APG · no clamp · ANC1",
        (
            ("Raw field · no anchor", "RAW_FLOWEDIT_ANC1_NO_ANCHOR_NO_CLAMP.mp4", "One target and one source conditional velocity per cell"),
            ("Raw field · balanced TR-QK early 3", "RAW_TRQK_A010_EARLY3_NO_CLAMP.mp4", "Pure-T2V route enters the sole target conditional call"),
        ),
    ),
    (
        "Round 7 · source-free T2V generation field · raw · no clamp · ANC1",
        (
            ("T2V field · no anchor", "T2V_FIELD_ANC1_NO_ANCHOR.mp4", "Source/target captions query the generation field"),
            ("T2V field · online anchor early 3", "T2V_FIELD_TRQK_A010_EARLY3.mp4", "Full-sequence anchor Q/K route in target generation call"),
        ),
    ),
    (
        "Round 8 · stable raw RV2V field · stronger action route · ANC1",
        (
            ("Raw TR-QK α.50 · early 8", "RAW_TRQK_A050_EARLY8.mp4", "Five-times stronger temporal route on stable field"),
            ("Raw hard-K · early 3", "RAW_HARDK_EARLY3.mp4", "Strong routing probe; current Q/V retained"),
        ),
    ),
    (
        "Round 9 · T2V source/target/anchor role split · raw · no clamp · ANC1",
        (
            ("Role-split T2V field · no anchor", "T2V_ROLESPLIT_ANC1_NO_ANCHOR.mp4", "Target caption preserves source appearance and adds only the action"),
            ("Role-split T2V field · online anchor early 3", "T2V_ROLESPLIT_TRQK_A010_EARLY3.mp4", "Anchor appearance is visible only to the separate anchor forward"),
        ),
    ),
    (
        "Round 10 · stable raw RV2V field · temporal value transport · ANC1",
        (
            ("Raw TR-QKV α.25 · early 8", "RAW_TRQKV_A025_EARLY8.mp4", "Sparse frame-relative Q/K/V residual; no absolute anchor V"),
            ("Raw TR-V α.50 · early 8", "RAW_TRV_A050_EARLY8.mp4", "Only sparse frame-relative value residual is transported"),
        ),
    ),
    (
        "Round 11 · temporal value transport · legacy τ=1 SGA control",
        (
            ("τ=1 SGA · TR-QKV α.25", "SGA5_RAW_TRQKV_A025_EARLY8.mp4", "Diagnostic only: legacy temperature makes close cosine scores nearly uniform"),
            ("τ=1 SGA · TR-V α.50", "SGA5_RAW_TRV_A050_EARLY8.mp4", "Diagnostic only: effectively close to AVG5 on source-initialized latents"),
        ),
    ),
    (
        "Round 12 · post-attention aggregate transport",
        (
            ("ANC1 · TR-attn-output α.50", "ANC1_RAW_TRO_A050_EARLY8.mp4", "Frame-relative residual after attention integrates Q/K/V"),
            ("τ=1 SGA control · TR-attn-output α.50", "SGA5_RAW_TRO_A050_EARLY8.mp4", "Legacy near-uniform SGA control; not the final DynaEdit implementation"),
        ),
    ),
    (
        "Round 13 · DynaEdit τ=.01 · candidate-0 continuation diagnostic",
        (
            ("τ=.01 SGA · TR-attn-output α.50", "SGA001_RAW_TRO_A050_EARLY8.mp4", "Correct low-temperature selection; historical chain-0 continuation retained only as diagnostic"),
            ("τ=.01 SGA · full velocity residual α.25", "SGA001_RAW_TRVEL_A025_EARLY8.mp4", "Full pure-T2V velocity trajectory; historical chain-0 continuation diagnostic"),
        ),
    ),
    (
        "Round 14 · true SGA+ANC · τ=.01 + weighted chain collapse",
        (
            ("True SGA · TR-attn-output α.50", "SGA001C_RAW_TRO_A050_EARLY8.mp4", "Projected-clean low-temperature SGA and variance-normalized weighted ANC collapse"),
            ("True SGA · full velocity residual α.25", "SGA001C_RAW_TRVEL_A025_EARLY8.mp4", "Online pure-T2V full-field action route under complete SGA+ANC"),
        ),
    ),
    (
        "Round 15 · phase-0 semantic correspondence before attention transport",
        (
            ("True SGA · corr-attn-output · blocks 8–13", "SGA001C_CORRO_A050_EARLY3_B8_13.mp4", "Current tokens retrieve anchor temporal residuals by phase-0 V-feature correspondence"),
            ("True SGA · corr-attn-output · blocks 14–21", "SGA001C_CORRO_A050_EARLY3_B14_21.mp4", "Later semantic block band under the same correspondence route"),
        ),
    ),
    (
        "Round 16 · full-network velocity strength boundary · true SGA+ANC",
        (
            ("True SGA · velocity residual α.50", "SGA001C_TRVEL_A050_EARLY8.mp4", "Twice the Round-14 full-field route strength"),
            ("True SGA · velocity residual α1.00", "SGA001C_TRVEL_A100_EARLY8.mp4", "Hard upper bound for frame-relative sparse full-field transport"),
        ),
    ),
    (
        "Round 17 · DynaEdit clean state + source-initial-state field",
        (
            ("Static source phase · no anchor", "STATICFIELD_SGA001C_NOANCHOR.mp4", "Full source remains the clean edit state; the visual field receives only a static source-initial-state carrier"),
            ("Static source phase · velocity anchor", "STATICFIELD_SGA001C_TRVEL_A050_EARLY8.mp4", "Every active candidate queries the pure-T2V anchor and transports its full-field temporal residual"),
            ("Static source phase · correspondence attention", "STATICFIELD_SGA001C_CORRO_A050_EARLY3_B8_13.mp4", "Current source tokens retrieve anchor temporal attention residuals after the walking prefix is removed"),
            ("Source-free generation field · velocity anchor", "T2VFIELD_SGA001C_TRVEL_A050_EARLY8.mp4", "Action-generation upper bound; identity is judged separately"),
        ),
    ),
    (
        "Round 18 · online pure-T2V action-minus-noop contrast · all 40 steps",
        (
            ("Static first phase + caption field · no anchor", "CAPTIONSTATIC_SGA001C_NOANCHOR.mp4", "DynaEdit-like source/target full-caption velocity difference control"),
            ("Caption field + anchor contrast α.50", "CAPTIONSTATIC_SGA001C_CONTRAST_A050_ALL40.mp4", "Two anchor forwards per candidate: action minus appearance-matched no-op"),
            ("Editor field + anchor contrast α.50", "STATICFIELD_SGA001C_CONTRAST_A050_ALL40.mp4", "Same contrast route with in-distribution Bernini edit/no-op field prompts"),
            ("Editor field + correspondence attention α.25", "STATICFIELD_SGA001C_CORRO_A025_ALL40_B8_13.mp4", "Online block-level hard route remains active through the final solver cell"),
        ),
    ),
    (
        "Round 19 · action-minus-noop integrated attention hard route",
        (
            ("Caption field · attention contrast α.25 · all 40", "CAPTIONSTATIC_SGA001C_ATTCON_A025_ALL40_B8_13.mp4", "Two online anchor forwards per candidate; only their attention-output difference enters the target"),
            ("Caption field · attention contrast α.50 · early 8", "CAPTIONSTATIC_SGA001C_ATTCON_A050_EARLY8_B8_13.mp4", "Stronger route limited to coarse solver cells"),
            ("Editor field · attention contrast α.25 · all 40", "STATICFIELD_SGA001C_ATTCON_A025_ALL40_B8_13.mp4", "In-distribution edit/no-op outer field with all-step block contrast"),
            ("Editor field · attention contrast α.50 · early 8", "STATICFIELD_SGA001C_ATTCON_A050_EARLY8_B8_13.mp4", "In-distribution outer field with stronger coarse-only block contrast"),
        ),
    ),
    (
        "Round 20 · DynaEdit published source/target raw-velocity CFG",
        (
            ("CFG src2.5/tar4.5 · no anchor", "CAPTIONSTATIC_CFG25_45_SGA001C_NOANCHOR.mp4", "Published low-CFG DynaEdit control; source/target full captions and static source initial condition"),
            ("CFG src2.5/tar4.5 · attention contrast", "CAPTIONSTATIC_CFG25_45_SGA001C_ATTCON_A025_ALL40.mp4", "Low-CFG control plus online pure-T2V action-minus-noop block route"),
            ("CFG src4.5/tar8.5 · no anchor", "CAPTIONSTATIC_CFG45_85_SGA001C_NOANCHOR.mp4", "Published high-CFG configuration intended for large action deviation"),
            ("CFG src4.5/tar8.5 · attention contrast", "CAPTIONSTATIC_CFG45_85_SGA001C_ATTCON_A025_ALL40.mp4", "High-CFG control plus all-step online block action contrast"),
        ),
    ),
    (
        "Round 21 · DynaEdit raw CFG with Bernini editor prompt field",
        (
            ("Editor field · CFG src2.5/tar4.5 · no anchor", "STATICEDITOR_CFG25_45_SGA001C_NOANCHOR.mp4", "Low-CFG matched control with static source initial-state carrier"),
            ("Editor field · CFG src2.5/tar4.5 · attention contrast", "STATICEDITOR_CFG25_45_SGA001C_ATTCON_A025_ALL40.mp4", "Low-CFG editor field plus all-step online pure-T2V action contrast"),
            ("Editor field · CFG src4.5/tar8.5 · no anchor", "STATICEDITOR_CFG45_85_SGA001C_NOANCHOR.mp4", "High-CFG editor-field control for large action deviation"),
            ("Editor field · CFG src4.5/tar8.5 · attention contrast", "STATICEDITOR_CFG45_85_SGA001C_ATTCON_A025_ALL40.mp4", "High-CFG editor field plus all-step online block action contrast"),
        ),
    ),
    (
        "Round 22 · partial high-CFG action: CFG interpolation and sharper SGA",
        (
            ("Caption field · CFG src3.5/tar6.5 · no anchor", "CAPTIONSTATIC_CFG35_65_SGA001C_NOANCHOR.mp4", "Interpolated CFG tests whether object scale improves without losing crouch/grasp/lift"),
            ("Caption field · CFG src3.5/tar6.5 · attention contrast", "CAPTIONSTATIC_CFG35_65_SGA001C_ATTCON_A025_ALL40.mp4", "Matched online-anchor arm at the interpolated CFG"),
            ("Caption field · CFG src4.5/tar8.5 · SGA tau .001 · no anchor", "CAPTIONSTATIC_CFG45_85_SGA0001C_NOANCHOR.mp4", "Successful high CFG with a sharper source-similarity candidate choice"),
            ("Caption field · CFG src4.5/tar8.5 · SGA tau .001 · attention contrast", "CAPTIONSTATIC_CFG45_85_SGA0001C_ATTCON_A025_ALL40.mp4", "Matched all-step anchor route under sharper SGA"),
        ),
    ),
    (
        "Round 23 · pure-T2V anchor as target-conditional block guidance",
        (
            ("Conditional anchor alpha .03 · early 8", "CAPTIONSTATIC_CFG45_85_CONDATT_A003_EARLY8.mp4", "Anchor contrast enters only target conditional forwards; effective CFG-scaled strength is about .255"),
            ("Conditional anchor alpha .03 · all 40", "CAPTIONSTATIC_CFG45_85_CONDATT_A003_ALL40.mp4", "Same conditional-only route through the endpoint"),
            ("Conditional anchor alpha .06 · early 8", "CAPTIONSTATIC_CFG45_85_CONDATT_A006_EARLY8.mp4", "Stronger coarse-only conditional action route; effective scale is about .51"),
            ("Conditional anchor alpha .06 · all 40", "CAPTIONSTATIC_CFG45_85_CONDATT_A006_ALL40.mp4", "Stronger route remains active in coarse and detail solver cells"),
        ),
    ),
    (
        "Round 24 · precise target caption for the source object instance",
        (
            ("Precise small-pebble caption · CFG src4.5/tar8.5 · no anchor", "PRECISEOBJ_CFG45_85_NOANCHOR.mp4", "Names the loose grey pebble and explicitly freezes every large white stepping stone"),
            ("Precise caption · CFG src4.5/tar8.5 · conditional anchor", "PRECISEOBJ_CFG45_85_CONDATT_A003_EARLY8.mp4", "Matched target-conditional pure-T2V route at the partial high-CFG action point; same-stone contact and trajectory remain strict review criteria"),
            ("Precise small-pebble caption · CFG src4.0/tar7.5 · no anchor", "PRECISEOBJ_CFG40_75_NOANCHOR.mp4", "Tests an action-strength point between the incomplete and oversized-object regimes"),
            ("Precise caption · CFG src4.0/tar7.5 · conditional anchor", "PRECISEOBJ_CFG40_75_CONDATT_A003_EARLY8.mp4", "Matched target-conditional anchor at the intermediate-high CFG"),
        ),
    ),
    (
        "Round 25 · DynaEdit SGA / ANC causal ablation",
        (
            ("IID single candidate · original caption", "CORE_IID1_ORIGCAPTION.mp4", "Neither SGA nor ANC; isolates high-CFG vanilla FlowEdit behavior"),
            ("ANC single candidate · original caption", "CORE_ANC1_ORIGCAPTION.mp4", "Adds only annealed cross-step noise correlation"),
            ("Uniform five candidates + ANC · original caption", "CORE_AVG5_ORIGCAPTION.mp4", "Matches SGA compute without source-similarity weighting"),
            ("SGA + ANC · compact small-pebble caption", "COMPACTOBJ_SGA5.mp4", "Short object disambiguation while keeping action verbs dominant"),
        ),
    ),
    (
        "Round 26 · pure-T2V action contrast at text cross-attention",
        (
            ("Cross-attention contrast α.03 · early 8 · blocks 4–9", "CROSSATT_A003_EARLY8_B4_9.mp4", "Action-minus-noop pure-T2V attn2 output enters target conditional rows only"),
            ("Cross-attention contrast α.03 · early 8 · blocks 10–15", "CROSSATT_A003_EARLY8_B10_15.mp4", "Separates an earlier/middle semantic block band"),
            ("Cross-attention contrast α.06 · early 8 · blocks 4–9", "CROSSATT_A006_EARLY8_B4_9.mp4", "Stronger coarse-stage text-action route"),
            ("Cross-attention contrast α.03 · all 40 · blocks 4–9", "CROSSATT_A003_ALL40_B4_9.mp4", "Tests whether the cross-attention action signal is needed through the endpoint"),
        ),
    ),
    (
        "Round 27 · actual anchor-video motion: dynamic minus static",
        (
            ("Dynamic−static velocity α.10 · early 8", "DYNSTATIC_VEL_A010_EARLY8.mp4", "Full-network pure-T2V difference under the same action caption and noise"),
            ("Dynamic−static velocity α.25 · early 8", "DYNSTATIC_VEL_A025_EARLY8.mp4", "Stronger full-field anchor-video motion route"),
            ("Dynamic−static self-attention α.03 · early 8", "DYNSTATIC_SELFATT_A003_EARLY8_B8_13.mp4", "Block-local visual attention difference from dynamic versus phase-0-static anchor"),
            ("Dynamic−static cross-attention α.03 · early 8", "DYNSTATIC_CROSSATT_A003_EARLY8_B4_9.mp4", "Text cross-attention response to dynamic versus static anchor under identical caption"),
        ),
    ),
    (
        "Round 28 R1 · INVALID: capped latent used the unmatched outer timestep",
        (
            ("INVALID · velocity α.10 · latent cap .8", "DYNSTATIC_VEL_A010_EARLY8_CAP08.mp4", "Do not use for method judgment: sigma_anchor=.8 was paired with the outer timestep"),
            ("INVALID · velocity α.10 · latent cap .6", "DYNSTATIC_VEL_A010_EARLY8_CAP06.mp4", "Do not use for method judgment: sigma_anchor=.6 was paired with the outer timestep"),
            ("INVALID · self-attention α.03 · latent cap .8", "DYNSTATIC_SELFATT_A003_EARLY8_B8_13_CAP08.mp4", "Off-manifold teacher coordinate; retained only as an engineering artifact"),
            ("INVALID · cross-attention α.03 · latent cap .8", "DYNSTATIC_CROSSATT_A003_EARLY8_B4_9_CAP08.mp4", "Off-manifold teacher coordinate; retained only as an engineering artifact"),
        ),
    ),
    (
        "Round 29 · DynaEdit SGA/ANC plus hard visual-attention routing",
        (
            ("Hard target Q/K · cap .8 · first 1 · blocks 4–9", "HARD_QK_CAP08_FIRST1_B4_9.mp4", "Replace target Q/K only at the first coarse cell; retain current target/source V"),
            ("Hard target Q/K · cap .8 · first 3 · blocks 4–9", "HARD_QK_CAP08_FIRST3_B4_9.mp4", "Hard anchor attention topology across all five-candidate SGA cells"),
            ("Hard target K · cap .8 · first 3 · blocks 4–9", "HARD_K_CAP08_FIRST3_B4_9.mp4", "Keep current target queries and values; replace only anchor keys"),
            ("Dynamic−static Q/K α.10 · cap .8 · early 8", "DYNSTATIC_QK_A010_CAP08_EARLY8_B4_9.mp4", "Sparse Q/K contrast before attention; no anchor value stream is copied"),
        ),
    ),
    (
        "Round 30 · dense source-motion preservation support",
        (
            ("Source support top 10% + dilation 1 · hard outside", "SRCMOTION_K10_D1_HARD.mp4", "Only the source moving-subject/contact neighborhood may depart from the source latent"),
            ("Source support top 20% + dilation 1 · hard outside", "SRCMOTION_K20_D1_HARD.mp4", "Wider editable support; frame 0 and all outside tokens remain exact source"),
            ("Source support top 30% + dilation 1 · hard outside", "SRCMOTION_K30_D1_HARD.mp4", "Tests whether the full crouch and contact need a broader source-derived region"),
            ("Source support top 20% + dilation 1 · outside 5%", "SRCMOTION_K20_D1_SOFT05.mp4", "Allows a small global update to reduce hard-mask boundary artifacts"),
        ),
    ),
    (
        "Round 31 · sparse contacted-object residual support",
        (
            ("Source top 10% + residual .5%", "SRCMOTION_K10_RESID005.mp4", "Per-phase top .5% target displacement outside source motion may move an initially static object"),
            ("Source top 10% + residual 1%", "SRCMOTION_K10_RESID010.mp4", "Slightly wider sparse contact/object trajectory while the rest stays exact source"),
            ("Source top 10% + residual 2%", "SRCMOTION_K10_RESID020.mp4", "Tests the transition from a small movable object to renewed object expansion"),
            ("Source top 20% + residual 1%", "SRCMOTION_K20_RESID010.mp4", "Broader actor support paired with the same narrow contacted-object allowance"),
        ),
    ),
    (
        "Round 32 · object-size threshold plus paired pure-T2V anchor",
        (
            ("Residual 1.25% · no anchor", "SRCMOTION_K10_RESID0125_NOANCHOR.mp4", "Fine object-support control between the empty-hand 1% and oversized-object 2% arms"),
            ("Residual 1.5% · no anchor", "SRCMOTION_K10_RESID015_NOANCHOR.mp4", "Matched preservation control for the two active-anchor arms"),
            ("Residual 1.5% + dynamic−static Q/K anchor", "SRCMOTION_K10_RESID015_DYNQK.mp4", "Online pure-T2V Q/K contrast; target/source V and dense preservation remain authoritative"),
            ("Residual 1.5% + dynamic−static velocity anchor", "SRCMOTION_K10_RESID015_DYNVEL.mp4", "Online full-field anchor contrast under the same object-size constraint"),
        ),
    ),
    (
        "Round 33 · full-path pure-T2V anchor inside DynaEdit SGA/ANC",
        (
            ("40/40 hard Q/K · blocks 4–9", "FULLPATH_HARD_QK_B4_9.mp4", "Pure-T2V Q/K replaces target Q/K at every solver step/candidate; target V remains current"),
            ("40/40 hard K · blocks 4–9", "FULLPATH_HARD_K_B4_9.mp4", "Pure-T2V K replaces target K at every solver step/candidate; target Q/V remain current"),
            ("40/40 dynamic−static Q/K · .25", "FULLPATH_DYNQK_A025_B4_9.mp4", "Action-minus-static Q/K intervention at every solver step/candidate"),
            ("40/40 dynamic−static attention output · .10", "FULLPATH_DYNATTN_A010_B4_9.mp4", "Action-minus-static block output intervention at every solver step/candidate"),
        ),
    ),
    (
        "Round 34 · background-masked SGA + annealed source preservation",
        (
            ("Global SGA · preserve step 8→15", "FULLPATH_DYNATTN_GLOBALSGA_P8_R8.mp4", "Full-path anchor control; full-latent source cosine still ranks the first 15 candidates"),
            ("Background SGA · preserve step 8→15", "FULLPATH_DYNATTN_BGMASKSGA_P8_R8.mp4", "SGA audits only candidate-specific non-edit background; otherwise matched to the global arm"),
            ("Background SGA · preserve step 16→23", "FULLPATH_DYNATTN_BGMASKSGA_P16_R8.mp4", "Allows sixteen unconstrained solver updates before source authority ramps in"),
            ("Background SGA · preserve step 24→31", "FULLPATH_DYNATTN_BGMASKSGA_P24_R8.mp4", "Allows coarse and middle action/contact formation before late preservation"),
        ),
    ),
    (
        "Round 35 · phase-0 correspondence + dynamic/static block contrast",
        (
            ("Corresponded Q/K contrast · soft .25", "CORRCON_QK_A025_P16_R8.mp4", "Map target tokens to anchor phase-0 semantics before adding dynamic-minus-static Q/K"),
            ("Corresponded attention contrast · soft .25", "CORRCON_ATTN_A025_P16_R8.mp4", "Corresponded dynamic-minus-static route after attention aggregation"),
            ("Corresponded Q/K trajectory · hard", "CORRCON_HARD_QK_P16_R8.mp4", "Top-25% target Q/K trajectories are replaced; current phase-0 basis is retained"),
            ("Corresponded attention trajectory · hard", "CORRCON_HARD_ATTN_P16_R8.mp4", "Top-25% attention-output trajectories are replaced after correspondence"),
        ),
    ),
    (
        "Round 36 · mutual correspondence confidence gate",
        (
            ("Mutual Q/K · .50 · blocks 4–9", "MUTUALCORR_QK_A050_B4_9_P16_R8.mp4", "Route only bidirectionally consistent full-spatial token matches"),
            ("Mutual attention · .50 · blocks 4–9", "MUTUALCORR_ATTN_A050_B4_9_P16_R8.mp4", "Confidence-gated action contrast after early attention aggregation"),
            ("Mutual Q/K · .50 · blocks 10–15", "MUTUALCORR_QK_A050_B10_15_P16_R8.mp4", "Later semantic block band under the same mutual gate"),
            ("Mutual attention · .50 · blocks 10–15", "MUTUALCORR_ATTN_A050_B10_15_P16_R8.mp4", "Later attention-output route; matched outer DynaEdit state"),
        ),
    ),
    (
        "Round 37 · four diverse pure-T2V action candidates",
        (
            ("Anchor v0 · girl / white stone", "anchor.mp4", "Complete crouch-contact-lift-stand-hold; appearance is not target authority"),
            ("Anchor v1 · boy / grey stone", "anchor_v1.mp4", "Independent generation with a different person, scene and stone"),
            ("Anchor v2 · girl / dark flat stone", "anchor_v2.mp4", "Independent generation with a different scale and object geometry"),
            ("Anchor v3 · boy / reddish stone", "anchor_v3.mp4", "Independent generation with a fourth spatial action/object relation"),
        ),
    ),
    (
        "Round 37 · INVALID DynaEdit interpretation: anchor-bank aggregation",
        (
            ("INVALID · Bank4 SGA · background · velocity .10", "BANK4_SGA_BG_TRVEL_A010_P16_R8.mp4", "Not DynaEdit SGA: different anchor videos were bound to noise candidates and their latents were averaged"),
            ("INVALID · Bank4 uniform average · velocity .10", "BANK4_AVG_TRVEL_A010_P16_R8.mp4", "Diagnostic artifact only: exact 1/4 averaging of four unrelated anchor-video latents"),
            ("INVALID · Bank4 SGA · global · velocity .10", "BANK4_SGA_GLOBAL_TRVEL_A010_P16_R8.mp4", "Do not use to judge fixed-anchor SGA/ANC; candidate construction was wrong"),
            ("INVALID · Bank4 SGA · background · velocity .25", "BANK4_SGA_BG_TRVEL_A025_P16_R8.mp4", "Stronger route does not repair the invalid cross-anchor aggregation"),
        ),
    ),
    (
        "Round 38 · dense action-field spatial alignment",
        (
            ("Bank4 aligned velocity · .10", "BANK4_ALIGNED_SGA_BG_TRVEL_A010_P16_R8.mp4", "Full phase-space action route affine-aligned from each anchor motion support to source motion support"),
            ("Bank4 aligned velocity · .25", "BANK4_ALIGNED_SGA_BG_TRVEL_A025_P16_R8.mp4", "Matched stronger route; phase zero remains exact zero and source content is not replaced"),
        ),
    ),
    (
        "Round 39 · fixed-anchor SGA/ANC + coordinate-free hard temporal attention",
        (
            ("Fixed v0 · hard phase-mean Q/K · blocks 0–5", "FIXEDV0_PHASEMEAN_QK_B0_5_P16_R8.mp4", "One anchor in every model cell; SGA explores five noise paths, never different anchors; target V/content retained"),
            ("Fixed v0 · hard phase-mean attention · blocks 0–5", "FIXEDV0_PHASEMEAN_ATTN_B0_5_P16_R8.mp4", "Hard-replace the coordinate-free temporal component after early attention aggregation"),
            ("Fixed v0 · hard phase-mean Q/K · blocks 4–9", "FIXEDV0_PHASEMEAN_QK_B4_9_P16_R8.mp4", "Matched middle-early block band; target spatial residual and phase zero remain source-owned"),
            ("Fixed v0 · hard phase-mean attention · blocks 4–9", "FIXEDV0_PHASEMEAN_ATTN_B4_9_P16_R8.mp4", "Matched output seam with full-path dynamic/static anchor forwards"),
        ),
    ),
    (
        "Round 40 · fixed-anchor pre-RoPE hidden route → native Q/K projection",
        (
            ("Fixed v0 · pre-RoPE phase-mean → Q/K · blocks 0–5", "FIXEDV0_PREROPE_PHASEMEAN_QK_B0_5_P16_R8.mp4", "Gather full Ulysses hidden sequence, hard-route temporal mean, reapply native Q/K projection+RoPE, retain original target V"),
            ("Fixed v0 · pre-RoPE phase-mean → Q/K · blocks 4–9", "FIXEDV0_PREROPE_PHASEMEAN_QK_B4_9_P16_R8.mp4", "Matched middle-early block stop gate; one fixed anchor and five SGA noise paths"),
        ),
    ),
    (
        "Round 41 · fixed-anchor dynamic/static temporal-attention kernel",
        (
            ("Fixed v0 · temporal kernel → target V · blocks 0–5", "FIXEDV0_TEMPORAL_KERNEL_ATTN_B0_5_P16_R8.mp4", "Per-head 21×21 dynamic-minus-static anchor attention kernel; applied only to target V; no anchor feature or value copied"),
            ("Fixed v0 · temporal kernel → target V · blocks 4–9", "FIXEDV0_TEMPORAL_KERNEL_ATTN_B4_9_P16_R8.mp4", "Matched middle-early band with the same fixed anchor, SGA noise candidates and ANC chain"),
        ),
    ),
    (
        "Round 42 · pure-T2V anchor action reward inside DynaEdit SGA",
        (
            ("Background + .02 anchor-action reward", "ANCHOR_ACTION_SGA_ADD002_P16_R8.mp4", "Anchor ranks complete projected-clean candidates by local 21×21 temporal self-similarity; no feature transport"),
            ("Background trust .003 → anchor-action reward", "ANCHOR_ACTION_SGA_TRUST003_P16_R8.mp4", "Only candidates within .003 of best background preservation remain eligible; anchor action chooses among them"),
        ),
    ),
    (
        "Round 43 · canonical dense motion-envelope reward inside SGA",
        (
            ("Background + .05 anchor envelope", "ANCHOR_ENVELOPE_SGA_ADD005_P16_R8.mp4", "Canonical 21×16×16 temporal-derivative energy envelope retains action direction and support; no feature injection"),
            ("Background trust .003 → anchor envelope", "ANCHOR_ENVELOPE_SGA_TRUST003_P16_R8.mp4", "Preservation-feasible candidates are ranked by the same dense anchor motion envelope"),
        ),
    ),
    (
        "Round 44 · fixed reward, expanded DynaEdit SGA proposal pool",
        (
            ("Anchor envelope trust .003 · 8 proposals", "ANCHOR_ENVELOPE_SGA_TRUST003_K8_P16_R8.mp4", "The same fixed v0 action anchor ranks 8 independent noise proposals at each of the first three coarse solver cells"),
            ("Anchor envelope trust .003 · 12 proposals", "ANCHOR_ENVELOPE_SGA_TRUST003_K12_P16_R8.mp4", "Matched arm with 12 early proposals; this is proposal diversity, not multi-anchor retrieval"),
        ),
    ),
    (
        "Round 45 · online anchor + target-gated hard temporal topology",
        (
            ("Hard temporal kernel · target top 10% · all 40", "TARGETGATED_HARDKERNEL_TOP10_ALL40_B4_9_P16_R8.mp4", "Dynamic/static pure-T2V anchor is forwarded in every SGA/ANC cell; hard replacement touches only the target's top-10% active sites and uses target V"),
            ("Hard temporal kernel · target top 25% · all 40", "TARGETGATED_HARDKERNEL_TOP25_ALL40_B4_9_P16_R8.mp4", "Matched wider target-local gate; phase 0 and all unselected spatial sites are exact"),
        ),
    ),
    (
        "Round 46 · target-local hard topology only in coarse solver cells",
        (
            ("Target top 10% hard kernel · early 3", "TARGETGATED_HARDKERNEL_TOP10_EARLY3_B4_9_P16_R8.mp4", "Online dynamic/static anchor is active exactly during the three multi-candidate SGA cells; later target attention is untouched"),
            ("Target top 10% hard kernel · early 8", "TARGETGATED_HARDKERNEL_TOP10_EARLY8_B4_9_P16_R8.mp4", "Matched route extends five cells into the collapsed ANC chain, then stops before low-noise detail formation"),
        ),
    ),
    (
        "Round 47 · exact native anchor Gaussian as a DynaEdit SGA proposal",
        (
            ("Anchor native noise in candidate 0 · reward-selected", "ANCHOR_NATIVE_NOISE_SGA_TRUST003_P16_R8.mp4", "Step-0 candidate 0 uses the byte-audited Gaussian that generated anchor v0; SGA remains free to choose among five proposals"),
            ("Anchor native noise in candidate 0 · forced chain", "ANCHOR_NATIVE_NOISE_FORCED_C0_P16_R8.mp4", "Upper bound: the anchor-seeded candidate-0 ANC chain is forced through all three early aggregation cells"),
        ),
    ),
    (
        "Round 48 · native pure-T2V trajectories inside DynaEdit SGA/ANC blocks",
        (
            ("Native action/noop trajectory · additive attention .03 · early 3", "NATIVE_TRAJ_ADDATTN_A003_EARLY3_B8_13.mp4", "The audited Gaussian evolves through independent native APG+UniPC action/noop paths; their real step states supply an additive block contrast during all three SGA cells"),
            ("Native action/noop trajectory · hard phase mean · early 3", "NATIVE_TRAJ_HARDMEAN_EARLY3_B8_13.mp4", "Coordinate-free hard attention-output replacement from the native generation paths; SGA proposal noises remain independent"),
            ("Native trajectories · target top10% hard kernel · early 3", "NATIVE_TRAJ_HARDKERNEL_TOP10_EARLY3_B4_9.mp4", "Hard temporal topology is applied only to target-active sites during the three multi-candidate SGA solver cells"),
            ("Native trajectories · target top10% hard kernel · early 8", "NATIVE_TRAJ_HARDKERNEL_TOP10_EARLY8_B4_9.mp4", "Matched hard route extends five steps into the collapsed ANC chain; anchor seed never replaces an SGA noise candidate"),
        ),
    ),
    (
        "Round 49 · native pure-T2V action−noop velocity field inside DynaEdit SGA/ANC",
        (
            ("Native action−noop velocity · α.25 · first 3 SGA steps", "NATIVE_TRAJ_VEL_A025_EARLY3_AFFINE_P8_R8.mp4", "The exact action/no-op APG+UniPC trajectories are evaluated at each live timestep; their full packed velocity difference is sparsified, source-aligned and added to every target candidate"),
            ("Native action−noop velocity · α.50 · first 8 steps", "NATIVE_TRAJ_VEL_A050_EARLY8_AFFINE_P8_R8.mp4", "Matched stronger and longer field route; anchor Gaussian evolves only the two generation trajectories and never replaces an SGA proposal noise"),
        ),
    ),
    (
        "Round 50 · source-coordinate pure-T2V field with/without native trajectory phase gate",
        (
            ("Target-state T2V action−noop · α.50 · early 8", "TARGETSTATE_T2VCON_A050_EARLY8_BG_P8_R8.mp4", "Action/no-op APG is queried on every live target candidate, so its full velocity is already in source coordinates; the independently evolved native trajectory is a matched-compute control only"),
            ("Native phase-gated target-state T2V · α.50 · early 8", "NATIVEGATED_TARGETSTATE_T2VCON_A050_EARLY8_BG_P8_R8.mp4", "The same full target-state action field is modulated by the audited self-generated trajectory's 21-phase event-energy envelope; no anchor spatial tensor or appearance is copied"),
        ),
    ),
    (
        "Round 51 · native generation attention hard-routed inside DynaEdit SGA/ANC",
        (
            ("Native trajectory · hard Q/K · early 3", "NATIVE_TRAJ_HARDQK_EARLY3_B4_9_ENDPOINTSGA.mp4", "Exact pure-T2V APG+UniPC states supply uncompressed attn1 Q/K to target blocks 4–9; target V remains current-edit content; the saved endpoint also ranks early SGA proposals"),
            ("Native trajectory · hard K · early 3", "NATIVE_TRAJ_HARDK_EARLY3_B4_9_ENDPOINTSGA.mp4", "Matched conservative branch: only K is replaced by the native generation trajectory while target Q/V remain current-edit tensors; SGA/ANC and endpoint reward are identical"),
        ),
    ),
    (
        "Round 52 · anchor-Q early proposal + source-at-target-RoPE late K/V",
        (
            ("Anchor Q blocks 4–9 + source K/V all target sites blocks 18–29", "DUAL_ANCHORQ_EARLY_B4_9_SOURCEKV_LATE_B18_29_ALL.mp4", "Native T2V supplies only the early target query; late source hidden is reprojected with target RoPE and replaces every target K/V site"),
            ("Anchor Q blocks 4–9 + source K/V static 75% blocks 18–29", "DUAL_ANCHORQ_EARLY_B4_9_SOURCEKV_LATE_B18_29_STATIC75.mp4", "Matched route retains current target K/V only at the top-25% temporally active spatial sites; phase zero stays fully source-authoritative"),
        ),
    ),
    (
        "Round 53 diagnostic · near-native velocity probes (not exact path binding)",
        (
            ("Target-field replacement · early 3 · approximate", "NATIVE_T2V_TARGETVEL_REPLACE_EARLY3_ANCHORC0.mp4", "Diagnostic only: sigma[0] is 0.99999899 and source-contaminated target-state velocity differs from the native path; do not treat this as the exact-anchor result"),
            ("Generation action−noop delta · early 3 · approximate", "NATIVE_T2V_DELTAVEL_REPLACE_EARLY3_ANCHORC0.mp4", "Diagnostic only: step-0 candidate 0 was also rejected by SGA; retained only to expose why r2 was required"),
        ),
    ),
    (
        "Round 53r2 · exact native path bound to forced SGA candidate 0",
        (
            ("Exact target-field replacement · early 3", "NATIVE_PATHBOUND_TARGETVEL_REPLACE_EARLY3_FORCEDC0.mp4", "Native action/no-op velocities are bound at steps 0–2 with post-bind MSE exactly zero; SGA weights are forced to candidate 0"),
            ("Exact generation action−noop delta · early 3", "NATIVE_PATHBOUND_DELTAVEL_REPLACE_EARLY3_FORCEDC0.mp4", "Matched exact-path upper bound; both variants crouch early but return to the wrong branch-reaching attractor without stone contact"),
        ),
    ),
    (
        "Round 54 · exact native generation field across all 40 solver steps",
        (
            ("Full-path generation action−noop delta · no spatial preservation", "NATIVE_FULLPATH_DELTAVEL_REPLACE_ALL40_FORCEDC0_NOPRES.mp4", "Mechanistic action GO / quality NO-GO: crouch, stone pickup and rise appear, but yellow clothes, anchor stones and color artifacts overwrite the source distribution"),
            ("Full-path T2V target field − RV2V source field · no spatial preservation", "NATIVE_FULLPATH_TARGETVEL_REPLACE_ALL40_FORCEDC0_NOPRES.mp4", "Strict NO-GO: cross-distribution target/source subtraction collapses into high-frequency color noise and an anchor-yellow subject"),
        ),
    ),
    (
        "Round 55 · exact generation action−noop route cutoff sweep",
        (
            ("Generation delta cutoff · 8 steps", "NATIVE_DELTAVEL_CUTOFF08_FORCEDC0_NOPRES.mp4", "Identity mostly retained but action insufficient: the child returns to reaching toward the right-side branches without stone contact"),
            ("Generation delta cutoff · 16 steps", "NATIVE_DELTAVEL_CUTOFF16_FORCEDC0_NOPRES.mp4", "Transition point: late crouch begins, but lift/hold is incomplete and blue low-frequency contamination has already appeared"),
            ("Generation delta cutoff · 24 steps", "NATIVE_DELTAVEL_CUTOFF24_FORCEDC0_NOPRES.mp4", "Action stronger but source distribution collapsed into anchor stones, yellow clothing and blue/purple artifacts"),
            ("Generation delta cutoff · 32 steps", "NATIVE_DELTAVEL_CUTOFF32_FORCEDC0_NOPRES.mp4", "Monotonicity control: severe anchor-content takeover approaching the all-40 failure"),
        ),
    ),
    (
        "Round 56 · generation/RV2V handoff refinement",
        (
            ("Generation delta cutoff · 20 steps", "NATIVE_DELTAVEL_CUTOFF20_FORCEDC0_NOPRES.mp4", "No sweet point: crouch remains incomplete while blue field and yellow anchor ghost are already visible"),
            ("Generation delta cutoff · 22 steps", "NATIVE_DELTAVEL_CUTOFF22_FORCEDC0_NOPRES.mp4", "No sweet point: anchor-yellow actor becomes a duplicated ghost beside the source child before a source-bound lift is completed"),
        ),
    ),
    (
        "Round 57 · native generation temporal quotient hard replacement",
        (
            ("Full temporal action quotient · all 40", "NATIVE_TEMPORAL_QUOTIENT_FULL_ALL40_FORCEDC0_NOPRES.mp4", "NO-GO: phase-constant basis is removed, but time-varying yellow clothing, stones, occlusion and color artifacts still overwrite the source"),
            ("Top-25% sparse temporal action quotient · all 40", "NATIVE_TEMPORAL_QUOTIENT_SPARSE25_ALL40_FORCEDC0_NOPRES.mp4", "NO-GO: more source remains, but high-energy selection retains an independent yellow actor ghost and does not bind lift to the source child"),
        ),
    ),
    (
        "Round 58 · source-coordinate temporal action field with native timing",
        (
            ("Target-state full temporal quotient × native phase timing", "NATIVE_TIMED_TARGETSTATE_TEMPORAL_FULL_ALL40_FORCEDC0_NOPRES.mp4", "NO-GO: source identity and scene remain natural, but the child continues walking; native timing changes magnitude without supplying action/object direction"),
            ("Target-state sparse25 temporal quotient × native phase timing", "NATIVE_TIMED_TARGETSTATE_TEMPORAL_SPARSE25_ALL40_FORCEDC0_NOPRES.mp4", "NO-GO: matched sparse route is visually close to the full arm and still has no stop, crouch, stone contact, lift or hold"),
        ),
    ),
    (
        "Round 59 · raw source-coordinate action−noop field × native timing",
        (
            ("Raw target-state action−noop · full · no native timing", "TARGETSTATE_RAW_FULL_ALL40_FORCEDC0_NOPRES.mp4", "NO-GO matched control: source appearance remains natural but the child continues walking with no target-action stage"),
            ("Raw target-state action−noop · sparse25 · no native timing", "TARGETSTATE_RAW_SPARSE25_ALL40_FORCEDC0_NOPRES.mp4", "NO-GO matched top-25% control: retaining the absolute phase component still produces no stop, crouch, contact, lift or hold"),
            ("Raw target-state action−noop · full × native timing", "NATIVE_TIMED_TARGETSTATE_RAW_FULL_ALL40_FORCEDC0_NOPRES.mp4", "NO-GO: exact native timing changes the pixels but does not supply the missing source object/action direction"),
            ("Raw target-state action−noop · sparse25 × native timing", "NATIVE_TIMED_TARGETSTATE_RAW_SPARSE25_ALL40_FORCEDC0_NOPRES.mp4", "NO-GO: sparse raw route plus native timing remains the same source walking event"),
        ),
    ),
    (
        "Round 60 · five role proposals inside SGA/ANC（后验纠错：仅 proposal 1 明确落在 source 石头）",
        (
            ("Role-warp proposal5 · SGA · full field", "NATIVE_ROLEWARP_STONE5_SGA_FULL_P24_R8.mp4", "Coordinate audit correction: these are five generic role proposals, not five verified stones; only proposal 1 clearly overlaps the visible source stepping stone"),
            ("Role-warp proposal5 · uniform · full field", "NATIVE_ROLEWARP_STONE5_AVG_FULL_P24_R8.mp4", "Matched uniform candidate aggregation; proposal 0 continues after the early cells but is not a verified source-stone location"),
            ("Role-warp stone5 · SGA · sparse25", "NATIVE_ROLEWARP_STONE5_SGA_SPARSE25_P24_R8.mp4", "Matched SGA route with only each phase's top-25% role-warped field support"),
            ("Role-warp stone5 · uniform · sparse25", "NATIVE_ROLEWARP_STONE5_AVG_SPARSE25_P24_R8.mp4", "Matched uniform sparse control"),
        ),
    ),
    (
        "Round 61 · native actor-object attention graph on source-owned values",
        (
            ("Role graph · SGA stone5 · all 40", "NATIVE_ROLEGRAPH_SGA_ALL40_B4_9_P24_R8.mp4", "Native action/no-op Q/K defines a 21-phase actor-object graph inside blocks 4-9; five SGA candidates bind it to five source stones while all value/content features remain source-owned"),
            ("Role graph · uniform stone5 · all 40", "NATIVE_ROLEGRAPH_AVG_ALL40_B4_9_P24_R8.mp4", "Matched AVG5 control with identical native forwards, block intervention, preservation and compute; proposal 0 continues after the early cells"),
        ),
    ),
    (
        "Round 63 · additive native actor-object role-logit bias",
        (
            ("Role-logit bias · SGA · all 40", "NATIVE_ROLELOGIT_SGA_ALL40_B4_9_P24_R8.mp4", "Adds native action-minus-noop role-logit contrast to the target model's own role logits; preserves the existing target motion and source-owned values"),
            ("Role-logit bias · SGA · early 8", "NATIVE_ROLELOGIT_SGA_EARLY8_B4_9_P24_R8.mp4", "Matched coarse-step route; tests whether late graph injection disrupts the target model's already-correct crouch/lift motion"),
        ),
    ),
    (
        "Round 64 · phase-conditioned roles + source-object value carry",
        (
            ("Dynamic roles + source object · SGA", "NATIVE_DYNROLE_SOURCEOBJ_SGA_ALL40_B4_9_P24_R8.mp4", "Tracks anchor actor/object across 21 phases and routes only the selected source stone's phase-0 value along a target lift trajectory; SGA chooses the proposal"),
            ("Dynamic roles + source object · AVG", "NATIVE_DYNROLE_SOURCEOBJ_AVG_ALL40_B4_9_P24_R8.mp4", "Matched uniform candidate control; proposal 0 is retained after the early cells, exposing whether SGA selects the wrong source object"),
        ),
    ),
    (
        "Round 65 · dynamic source-object V route + native role graph",
        (
            ("Dynamic source-object V · α.50", "NATIVE_DYNROLE_OBJECTV_SGA_A050_ALL40_B4_9_P24_R8.mp4", "Moves the selected source stone's phase-0 V mean through the 21-phase target object path before attention; native action/no-op role-logit bias remains active"),
            ("Dynamic source-object V · α1.00", "NATIVE_DYNROLE_OBJECTV_SGA_A100_ALL40_B4_9_P24_R8.mp4", "Hard source-object V mean route; tests whether direct object identity transport can instantiate a continuous stone without leaving the video manifold"),
        ),
    ),
    (
        "Round 66 · source-object attention-output carry · SGA vs AVG",
        (
            ("Object output carry · SGA · α.50", "NATIVE_DYNROLE_OBJECTOUT_SGA_A050_ALL40_B4_9_P24_R8.mp4", "Native pure-T2V action/no-op Q/K supplies the relation graph; SGA selects proposal 2 while source-owned phase-0 object attention output is carried along its dynamic path"),
            ("Object output carry · uniform AVG · α.50", "NATIVE_DYNROLE_OBJECTOUT_AVG_A050_ALL40_B4_9_P24_R8.mp4", "Matched compute and route with uniform early aggregation and proposal 0 continuation; both arms remain natural but fail same-stone lift/hold"),
        ),
    ),
    (
        "Round 67 · observer-only native anchor · action reward comparison",
        (
            ("Observer only · action-Gram reward · P24/R8", "NATIVE_OBSERVER_SGA_ACTIONGRAM_EARLY3_B4_9_P24_R8.mp4", "Native action/no-op attention is captured and consumed but target output is bit-exact; top-25% local trajectory Gram/energy ranks candidates behind a source-background trust gate"),
            ("Observer only · motion-envelope reward · P24/R8", "NATIVE_OBSERVER_SGA_ENVELOPE_EARLY3_B4_9_P24_R8.mp4", "Matched candidate bank and compute; canonical motion envelope gives sharper weights, yet both rewards select no-object branch-reaching trajectories"),
        ),
    ),
    (
        "Round 68 · late-preservation causal ablation",
        (
            ("Observer only · action-Gram · P39/R8", "NATIVE_OBSERVER_SGA_ACTIONGRAM_EARLY3_B4_9_P39_R8.mp4", "Action recovers: ground contact, lift, stand and terminal hold. Object grounding still fails—the held object expands into an oversized grey boulder rather than preserving the source stepping-stone instance and scale"),
            ("Observer only · motion envelope · P39/R8", "NATIVE_OBSERVER_SGA_ENVELOPE_EARLY3_B4_9_P39_R8.mp4", "Matched late-preservation arm reaches the same causal conclusion: P24 had suppressed the interaction, but weakening it restores action at the cost of incorrect object identity/scale"),
        ),
    ),
    (
        "Round 69 · phase-wise source-object corridor · P24/R8",
        (
            ("Object-1 corridor · action-Gram", "NATIVE_OBSERVER_SGA_ACTIONGRAM_OBJECT1_P24_R8.mp4", "The audited source-object path is exempted from hard preservation while the rest of the scene keeps P24/R8. Lift returns and the boulder shrinks, but the flat source stepping stone is still redrawn as a rounded grey object"),
            ("Object-1 corridor · motion envelope", "NATIVE_OBSERVER_SGA_ENVELOPE_OBJECT1_P24_R8.mp4", "Matched corridor and source protection with the sharper envelope reward. Output remains close to action-Gram, isolating moving-object identity—not candidate ranking—as the next bottleneck"),
        ),
    ),
    (
        "Round 70 · translated source-object latent identity projection",
        (
            ("Object-1 identity projection · .025/step", "NATIVE_OBSERVER_SGA_ENVELOPE_OBJECT1_ID025_P24_R8.mp4", "Weak repeated projection preserves the action and scene but does not restore the flat source stepping-stone shape; it remains close to Round69"),
            ("Object-1 identity projection · .075/step", "NATIVE_OBSERVER_SGA_ENVELOPE_OBJECT1_ID075_P24_R8.mp4", "NO-GO: stronger clean-latent crop transport creates grey stamps, ghosting and temporal trails instead of coherent moving-object identity"),
        ),
    ),
    (
        "Round 71 · forced proposal-1 attention-level object authority",
        (
            ("Source object V before attention · α.50 · all40", "NATIVE_DYNROLE_OBJECTV_FORCEDP1_A050_B4_9_CORRIDOR_P24_R8.mp4", "Best attention arm: the child contacts and lifts a smaller grey object, and the object remains near the hand. Shape is still not the flat source stepping stone, so this is partial rather than final GO"),
            ("Source object attention output · α.50 · all40", "NATIVE_DYNROLE_OBJECTOUT_FORCEDP1_A050_B4_9_CORRIDOR_P24_R8.mp4", "Weaker branch: post-attention carry loses the object and returns to an empty-hand branch reach, showing identity authority must enter before target attention aggregation"),
        ),
    ),
    (
        "Round 72 · attention-V schedule and strength boundary",
        (
            ("Source object V · α.50 · early8", "NATIVE_DYNROLE_OBJECTV_FORCEDP1_A050_EARLY8_B4_9_CORRIDOR_P24_R8.mp4", "Early cutoff returns toward the larger rounded-object observer result and does not prevent the terminal hand/object path from entering the right-side foliage"),
            ("Source object V · α1.00 · all40", "NATIVE_DYNROLE_OBJECTV_FORCEDP1_A100_ALL40_B4_9_CORRIDOR_P24_R8.mp4", "Stronger V authority changes the trajectory but does not monotonically improve source-object shape; the same right/up terminal-path error remains"),
        ),
    ),
    (
        "Round 73 · anchor-relative terminal relation correction",
        (
            ("Corrected anchor relation · observer only", "NATIVE_ANCHORREL_PATH_OBSERVER_ENVELOPE_P24_R8.mp4", "Pure-T2V terminal object−actor relation replaces the former source-relative right/up path. The output changes materially but the target backbone still reaches into the foliage"),
            ("Corrected anchor relation · object V α.50 all40", "NATIVE_ANCHORREL_PATH_OBJECTV_FORCEDP1_A050_ALL40_P24_R8.mp4", "Corrected relation plus pre-attention source-object V; route and cache close correctly, yet α.50 does not overcome the same branch-reaching attractor"),
        ),
    ),
    (
        "Round 74 · corrected-relation hard attention upper bound",
        (
            ("Corrected object V · α1.00 · all40", "NATIVE_ANCHORREL_OBJECTV_FORCEDP1_A100_ALL40_P24_R8.mp4", "Hard pre-attention V changes contact details but still loses the object into the branch-reaching terminal attractor"),
            ("Corrected role graph + source object · α1.00 · all40", "NATIVE_ANCHORREL_ROLEGRAPH_SOURCEOBJ_FORCEDP1_A100_ALL40_P24_R8.mp4", "Hard joint relation/object route remains natural but does not maintain a visible held object; manual attention-strength sweep terminates here"),
        ),
    ),
    (
        "Round 75 · actor + object interaction corridor · final training-free stop gate",
        (
            ("Actor/object corridor · observer only", "NATIVE_ACTOROBJ_CORRIDOR_OBSERVER_ENVELOPE_P24_R8.mp4", "The phase-wise editable support now covers both the source-owned object path and the new crouch/arm-reach region. The child bends and rises naturally, but still reaches into the foliage without a continuous source-stone lift/hold"),
            ("Actor/object corridor · source object V α.50 all40", "NATIVE_ACTOROBJ_CORRIDOR_OBJECTV_FORCEDP1_A050_ALL40_P24_R8.mp4", "Pure-T2V action/no-op attention participates in every active target call and source-owned V is routed before attention. The result differs numerically from observer but does not bind the hand to the same source stone; this closes the manual-coordinate/strength branch"),
        ),
    ),
    (
        "Round 81 · target-suffix phase-0 2×2 patch move + online anchor relation graph",
        (
            ("Source authority", "source.mp4", "Only authority for the child, pale-blue clothing, exact stepping stones, garden, camera and frame 0"),
            ("Pure-T2V action anchor", "anchor.mp4", "Every active model call reads its action/no-op attention relation; anchor RGB, values and object appearance are never copied"),
            ("Round 68 action-positive control", "NATIVE_OBSERVER_SGA_ENVELOPE_EARLY3_B4_9_P39_R8.mp4", "Shows that SGA/ANC can select a complete lift, but the target model redraws the source stone as an oversized boulder"),
            ("Round 75 pooled-V control", "NATIVE_ACTOROBJ_CORRIDOR_OBJECTV_FORCEDP1_A050_ALL40_P24_R8.mp4", "Old carrier pools the source object to one vector and broadcasts it over an ellipse; action survives but same-stone scale/shape does not"),
            ("Target phase-0 patch · α1.00 · blocks 4–9", "NATIVE_SOURCEPATCH_MOVE_A100_ALL40_B4_9_ACTOROBJ_P24_R8.mp4", "Historical control: hard four-token pattern comes from the target-caption suffix's clamped phase zero, not the paired source branch; explicit vacancy and online action-minus-noop topology are active"),
            ("Target phase-0 patch · α0.50 · blocks 4–9", "NATIVE_SOURCEPATCH_MOVE_A050_ALL40_B4_9_ACTOROBJ_P24_R8.mp4", "Matched half-strength early-block control; the earlier source-owned label was incorrect and is fixed here"),
            ("Target phase-0 patch · α1.00 · blocks 8–13", "NATIVE_SOURCEPATCH_MOVE_A100_ALL40_B8_13_ACTOROBJ_P24_R8.mp4", "Hard target-suffix patch in deeper attention blocks; not an explicit paired-source carrier"),
            ("Target phase-0 patch · α0.50 · blocks 8–13", "NATIVE_SOURCEPATCH_MOVE_A050_ALL40_B8_13_ACTOROBJ_P24_R8.mp4", "Matched deeper half-strength control; no machine-positive label and no ground-truth target video"),
        ),
    ),
    (
        "Round 82 · target-suffix phase-0 2×2 patch in V before attention",
        (
            ("Source authority", "source.mp4", "Only authority for the child, pale-blue clothing, exact stepping stones, garden, camera and frame 0"),
            ("Pure-T2V action anchor", "anchor.mp4", "Online action/no-op Q/K relation is evaluated in every active model call; anchor values and appearance remain excluded"),
            ("Round 71 pooled-V control", "NATIVE_DYNROLE_OBJECTV_FORCEDP1_A050_B4_9_CORRIDOR_P24_R8.mp4", "Pre-attention pooled source V keeps a smaller object near the hand, but pooling destroys the flat stone's spatial pattern"),
            ("Round 81 post-output patch control", "NATIVE_SOURCEPATCH_MOVE_A100_ALL40_B4_9_ACTOROBJ_P24_R8.mp4", "Ordered four-token patch is injected after aggregation and is washed out into the foliage-reaching no-object event"),
            ("Target phase-0 patch V · α1.00 · blocks 4–9", "NATIVE_SOURCEPATCH_VALUE_A100_ALL40_B4_9_ACTOROBJ_P24_R8.mp4", "Historical control: hard ordered patch enters V from target-caption phase zero; current target Q/K and online anchor relation determine aggregation"),
            ("Target phase-0 patch V · α0.50 · blocks 4–9", "NATIVE_SOURCEPATCH_VALUE_A050_ALL40_B4_9_ACTOROBJ_P24_R8.mp4", "Matched half-strength early-block control; it is not the explicit paired-source route introduced in Round 84"),
            ("Target phase-0 patch V · α1.00 · blocks 8–13", "NATIVE_SOURCEPATCH_VALUE_A100_ALL40_B8_13_ACTOROBJ_P24_R8.mp4", "Hard target-suffix patch in deeper semantic attention blocks; all other causal coordinates remain fixed"),
            ("Target phase-0 patch V · α0.50 · blocks 8–13", "NATIVE_SOURCEPATCH_VALUE_A050_ALL40_B8_13_ACTOROBJ_P24_R8.mp4", "Matched deeper half-strength control; no machine-positive label and no target ground-truth video"),
        ),
    ),
    (
        "Round 83 · sparse source-clean entity state + target-suffix patch V",
        (
            ("Source authority", "source.mp4", "Only authority for the child, pale-blue clothing, exact stepping stones, garden, camera and frame 0"),
            ("Pure-T2V action anchor", "anchor.mp4", "Online action/no-op attention still supplies the action and contact relation; its appearance is excluded from the entity carrier"),
            ("Round 70 full-crop projection control", "NATIVE_OBSERVER_SGA_ENVELOPE_OBJECT1_ID075_P24_R8.mp4", "Historical clean-state projection copies a large rectangular neighborhood and can create stamps or trails; it is not the new sparse composite"),
            ("Round 82 patch-V control", "NATIVE_SOURCEPATCH_VALUE_A100_ALL40_B4_9_ACTOROBJ_P24_R8.mp4", "Attention-local source pattern changes the trajectory but is washed out before a visible same-stone lift"),
            ("Entity state · step .025 · blocks 4–9", "NATIVE_ENTITYSTATE_ID025_PATCHV_A100_ALL40_B4_9_P24_R8.mp4", "Compact ordered source signature is accumulated after ODE updates; 16 active applications give about .333 cumulative strength"),
            ("Entity state · step .075 · blocks 4–9", "NATIVE_ENTITYSTATE_ID075_PATCHV_A100_ALL40_B4_9_P24_R8.mp4", "Stronger matched early-block arm; about .713 cumulative entity strength with the same explicit origin vacancy"),
            ("Entity state · step .025 · blocks 8–13", "NATIVE_ENTITYSTATE_ID025_PATCHV_A100_ALL40_B8_13_P24_R8.mp4", "Matched deeper attention scope with the same sparse post-update source entity state"),
            ("Entity state · step .075 · blocks 8–13", "NATIVE_ENTITYSTATE_ID075_PATCHV_A100_ALL40_B8_13_P24_R8.mp4", "Stronger deeper arm; no machine-positive label and no target ground-truth video"),
        ),
    ),
    (
        "Round 84 · explicit paired source-branch carrier",
        (
            ("Source authority", "source.mp4", "The paired source suffix is now the actual phase-zero V/output carrier, rather than merely relying on the target suffix's clamped phase zero"),
            ("Pure-T2V action anchor", "anchor.mp4", "Every active call still captures action/no-op Q/K relations; anchor values, latent and RGB remain excluded"),
            ("Target-suffix patch-V control", "NATIVE_SOURCEPATCH_VALUE_A100_ALL40_B4_9_ACTOROBJ_P24_R8.mp4", "Round 82 reads its spatial signature from target-caption hidden phase zero and fails to produce a visible source stone"),
            ("Target-suffix + state control", "NATIVE_ENTITYSTATE_ID075_PATCHV_A100_ALL40_B4_9_P24_R8.mp4", "Round 83 adds the source clean-state sparse composite but retains the old attention carrier; shown only as a causal control"),
            ("Explicit source V · no state · blocks 4–9", "NATIVE_EXPLSRC_ID000_PATCHV_A100_ALL40_B4_9_P24_R8.mp4", "Only change from the Round 82 early-block arm is that the ordered V patch and vacancy fill come from the paired source branch"),
            ("Explicit source V · state .075 · blocks 4–9", "NATIVE_EXPLSRC_ID075_PATCHV_A100_ALL40_B4_9_P24_R8.mp4", "Combines explicit source-branch attention carrier with the preregistered sparse clean-like entity state"),
            ("Explicit source V · no state · blocks 8–13", "NATIVE_EXPLSRC_ID000_PATCHV_A100_ALL40_B8_13_P24_R8.mp4", "Matched deeper-block source-branch carrier without the clean-state composite"),
            ("Explicit source V · state .075 · blocks 8–13", "NATIVE_EXPLSRC_ID075_PATCHV_A100_ALL40_B8_13_P24_R8.mp4", "Matched deeper combined arm; no machine-positive label and no target ground-truth video"),
        ),
    ),
    (
        "Round 85 · full-block action relation and handedness quotient",
        (
            ("Source authority", "source.mp4", "The selected source stone begins on the screen-right side of the source actor; source identity, scene and frame zero remain authoritative"),
            ("Pure-T2V action anchor", "anchor.mp4", "The donor performs the action on the opposite screen side; lift progress, contact timing and relation magnitude are action information, while signed left/right is tested as a nuisance symmetry"),
            ("Action-positive / weak-preservation control", "NATIVE_OBSERVER_SGA_ENVELOPE_EARLY3_B4_9_P39_R8.mp4", "Round 68 reaches and visibly lifts on the source side, but produces an oversized object and scene drift"),
            ("P24 actor/object control", "NATIVE_ACTOROBJ_CORRIDOR_OBSERVER_ENVELOPE_P24_R8.mp4", "With identity preservation restored, the six-block observer follows the foliage-reaching empty-hand attractor"),
            ("Explicit source V · 6 blocks 4–9", "NATIVE_EXPLSRC_ID000_PATCHV_A100_ALL40_B4_9_P24_R8.mp4", "Round 84 exact signed anchor relation with a true source-branch carrier, used as the partial-coverage control"),
            ("Explicit source V · 6 blocks 8–13", "NATIVE_EXPLSRC_ID000_PATCHV_A100_ALL40_B8_13_P24_R8.mp4", "Matched deeper partial-coverage control"),
            ("Explicit source V · all 30 · exact anchor side", "NATIVE_EXPLSRC_ANCHORREL_PATCHV_A100_ALL40_ALL30_P24_R8.mp4", "Hard upper bound: every transformer block receives the donor's exact signed terminal relation and explicit source-branch patch V"),
            ("Explicit source V · all 30 · source-side aligned", "NATIVE_EXPLSRC_SIDEALIGN_PATCHV_A100_ALL40_ALL30_P24_R8.mp4", "Same full-block upper bound, but horizontal relation is reflected only when donor and source object/actor ordering disagree; temporal progress and relation magnitude are unchanged"),
        ),
    ),
    (
        "Round 86 · full-block late-preservation test",
        (
            ("Source authority", "source.mp4", "Identity, pale-blue clothing, the original stepping stones, garden, camera and frame zero are judged only against this video"),
            ("Pure-T2V action anchor", "anchor.mp4", "Online action/no-op trajectory and attention relation are active; anchor appearance is never an identity target"),
            ("Observer · P39 action-positive control", "NATIVE_OBSERVER_SGA_ENVELOPE_EARLY3_B4_9_P39_R8.mp4", "A lighter 3-step observer keeps a visible but oversized object through the endpoint when preservation begins only at the final solver step"),
            ("Observer · P24 empty-hand control", "NATIVE_ACTOROBJ_CORRIDOR_OBSERVER_ENVELOPE_P24_R8.mp4", "Earlier preservation erases the transient object before the endpoint, motivating a matched preservation-time bisection"),
            ("Explicit source V · all 30 · exact · P24", "NATIVE_EXPLSRC_ANCHORREL_PATCHV_A100_ALL40_ALL30_P24_R8.mp4", "Round 85 full-depth hard transport with preservation beginning at step 24; no same source stone reaches the hand"),
            ("Explicit source V · all 30 · exact · P32", "NATIVE_EXPLSRC_ANCHORREL_PATCHV_A100_ALL40_ALL30_P32_R8.mp4", "Only the preservation start changes from P24 to P32; the child still ends by reaching empty-handed into foliage"),
            ("Rejected symmetry quotient · P24", "NATIVE_EXPLSRC_SIDEALIGN_PATCHV_A100_ALL40_ALL30_P24_R8.mp4", "Horizontal relation reflection destroys the subject and foreground, so this is a negative control rather than a candidate method"),
            ("Rejected symmetry quotient · P32", "NATIVE_EXPLSRC_SIDEALIGN_PATCHV_A100_ALL40_ALL30_P32_R8.mp4", "Later preservation does not rescue the reflected route; widespread colored fragments independently confirm the quotient is invalid"),
        ),
    ),
    (
        "Round 87 · matched preservation boundary",
        (
            ("Source authority", "source.mp4", "Judge the child's identity, blue clothing, exact stone instances, garden, camera and frame zero only against this video"),
            ("Pure-T2V action anchor", "anchor.mp4", "Supplies online action/no-op trajectory, SGA action reward and, in explicit arms, block attention relation; its appearance is not copied"),
            ("Observer · P39 reference", "NATIVE_OBSERVER_SGA_ENVELOPE_EARLY3_B4_9_P39_R8.mp4", "Only one weak first-ramp preservation application; a visible but oversized stone persists to the endpoint"),
            ("P24 actor/object reference", "NATIVE_ACTOROBJ_CORRIDOR_OBSERVER_ENVELOPE_P24_R8.mp4", "A transient object appears near contact but full late projection removes it before the endpoint"),
            ("Observer · P32 · R8", "NATIVE_OBSERVER_ENVELOPE_B4_9_P32_R8.mp4", "Round68-matched 3-step observer; all eight preservation levels run and the transient stone disappears before frame 80"),
            ("Observer · P36 · R8", "NATIVE_OBSERVER_ENVELOPE_B4_9_P36_R8.mp4", "Only four preservation levels run, ending at outside scale .525; a stone is visible near frame 40 but is still erased after frame 60"),
            ("Explicit source V · P32 · R8", "NATIVE_EXPLSRC_ANCHORREL_PATCHV_A100_ALL40_B4_9_P32_R8.mp4", "Six-block explicit paired-source carrier makes the lifted stone clearer through frames 30–50, but the complete preservation ramp removes it"),
            ("Explicit source V · P36 · R8", "NATIVE_EXPLSRC_ANCHORREL_PATCHV_A100_ALL40_B4_9_P36_R8.mp4", "Matched four-level preservation arm; clear mid-video object/contact still collapses into foliage before the terminal state"),
        ),
    ),
    (
        "Round 88 · terminal projection boundary",
        (
            ("Source authority", "source.mp4", "Only source identity, clothing, stones, garden, camera and frame zero are authoritative"),
            ("Pure-T2V action anchor", "anchor.mp4", "Supplies the action/no-op relation online; donor appearance is deliberately non-authoritative"),
            ("P39 reference · one weak projection", "NATIVE_OBSERVER_SGA_ENVELOPE_EARLY3_B4_9_P39_R8.mp4", "A large object survives to the endpoint, showing that the action route can create and retain a terminal object when preservation is nearly absent"),
            ("Observer · P37 · three projections", "NATIVE_OBSERVER_ENVELOPE_B4_9_P37_R8.mp4", "The object is visible through the contact/lift phase but disappears after frame 60"),
            ("Explicit source V · P37 · three projections", "NATIVE_EXPLSRC_ANCHORREL_PATCHV_A100_ALL40_B4_9_P37_R8.mp4", "The paired source carrier strengthens the mid-event object, yet the same terminal projection schedule still erases it"),
            ("Observer · P38 · two projections", "NATIVE_OBSERVER_ENVELOPE_B4_9_P38_R8.mp4", "Even only two weak late source projections delete the interaction object before the endpoint"),
            ("Explicit source V · P38 · two projections", "NATIVE_EXPLSRC_ANCHORREL_PATCHV_A100_ALL40_B4_9_P38_R8.mp4", "Matched hard-carrier boundary; judge terminal object persistence rather than machine reward"),
            ("P36 reference · four projections", "NATIVE_OBSERVER_ENVELOPE_B4_9_P36_R8.mp4", "More aggressive late preservation is included to make the monotonic deletion failure directly visible"),
        ),
    ),
    (
        "Round 90 · fixed interaction-support snapshot",
        (
            ("Source authority", "source.mp4", "Identity, blue clothing, exact stones, garden, camera and frame zero remain source-owned"),
            ("Pure-T2V action anchor", "anchor.mp4", "Online action/no-op relation only; donor appearance is not an identity or object target"),
            ("Frozen RV2V", "frozen.mp4", "Matched failure baseline; the new mode is not optimized to stay near this output"),
            ("P39 · one weak projection", "NATIVE_OBSERVER_SGA_ENVELOPE_EARLY3_B4_9_P39_R8.mp4", "Historical action-positive control with terminal oversized object and weaker scene preservation"),
            ("Snapshot · P32 · RF.015", "NATIVE_OBSERVER_SNAPSHOT_P32_RF015_R8.mp4", "Freeze the first top-1.5% per-phase residual support, then run all eight preservation levels"),
            ("Snapshot · P36 · RF.015", "NATIVE_OBSERVER_SNAPSHOT_P36_RF015_R8.mp4", "Same sparse support size; snapshot later and apply only four preservation levels"),
            ("Snapshot · P32 · RF.020", "NATIVE_OBSERVER_SNAPSHOT_P32_RF020_R8.mp4", "Matched earlier snapshot with a top-2% interaction residual support"),
            ("Snapshot · P36 · RF.020", "NATIVE_OBSERVER_SNAPSHOT_P36_RF020_R8.mp4", "Matched later snapshot and larger sparse support; judge object chain before appearance"),
        ),
    ),
    (
        "Round 91 · final-only strong source projection",
        (
            ("Source authority", "source.mp4", "Judge identity, clothing, stone instances, garden, camera and initial state only against source"),
            ("Pure-T2V action anchor", "anchor.mp4", "Online action/no-op donor; its appearance remains excluded from final identity"),
            ("Frozen RV2V", "frozen.mp4", "Matched action-failure baseline, not the target of a small-difference loss"),
            ("P39 · weak final projection", "NATIVE_OBSERVER_SGA_ENVELOPE_EARLY3_B4_9_P39_R8.mp4", "One final projection at outside scale .88125 retains an oversized object but permits scene drift"),
            ("Final strong · K.10 · RF.015", "NATIVE_OBSERVER_FINALSTRONG_K10_RF015.mp4", "No preservation before the final update; then one scale-.05 projection with stricter source-motion support"),
            ("Final strong · K.10 · RF.020", "NATIVE_OBSERVER_FINALSTRONG_K10_RF020.mp4", "Matched stricter background support with a slightly larger final interaction residual set"),
            ("Final strong · K.20 · RF.015", "NATIVE_OBSERVER_FINALSTRONG_K20_RF015.mp4", "Larger source-motion support; still only one final projection and no later model call"),
            ("Final strong · K.20 · RF.020", "NATIVE_OBSERVER_FINALSTRONG_K20_RF020.mp4", "Most permissive matched final-fusion arm; same seed, SGA/ANC and action route"),
        ),
    ),
    (
        "Round 92 · explicit actor-object final corridor",
        (
            ("Source authority", "source.mp4", "Source owns child identity, blue clothing, exact stones, garden, camera and frame zero"),
            ("Pure-T2V action anchor", "anchor.mp4", "Provides online action/no-op relation; donor appearance is not copied as identity"),
            ("P39 weak-projection reference", "NATIVE_OBSERVER_SGA_ENVELOPE_EARLY3_B4_9_P39_R8.mp4", "Retains an oversized terminal object but leaves scene drift"),
            ("Final strong generic support", "NATIVE_OBSERVER_FINALSTRONG_K20_RF020.mp4", "Round 91 negative control: generic residual support erases the object at the final projection"),
            ("Corridor observer · ID0", "NATIVE_FINALCORRIDOR_OBSERVER_ID000.mp4", "Explicit proposal1-to-hand corridor bypasses final source projection; no source-stone identity injection"),
            ("Corridor observer · ID.025", "NATIVE_FINALCORRIDOR_OBSERVER_ID025.mp4", "Same observer route plus a weak sparse source-stone signature at the final state"),
            ("Corridor explicit source · ID0", "NATIVE_FINALCORRIDOR_EXPLICIT_ID000.mp4", "All-40 paired-source patch-V carrier with the same final corridor and no identity injection"),
            ("Corridor explicit source · ID.025", "NATIVE_FINALCORRIDOR_EXPLICIT_ID025.mp4", "Combined source carrier, final corridor and weak stone signature; strict same-instance evidence required"),
        ),
    ),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build(root: Path) -> None:
    media = root / "media"
    if not media.is_dir():
        raise SystemExit("media directory is absent")
    manifest = {"schema_version": "anchor-sga-anc-event01-review-v1", "sections": []}
    section_markup = []
    for section_index, (title, cards) in enumerate(SECTIONS):
        rows = []
        cards_markup = []
        for label, filename, note in cards:
            path = media / filename
            if not path.is_file():
                continue
            rows.append(
                {
                    "label": label,
                    "path": f"media/{filename}",
                    "note": note,
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
            cards_markup.append(
                f'''<article class="card">
  <h3>{html.escape(label)}</h3>
  <video controls muted loop playsinline preload="metadata" data-group="g{section_index}" src="media/{html.escape(filename)}"></video>
  <p>{html.escape(note)}</p>
</article>'''
            )
        if not rows:
            continue
        manifest["sections"].append({"title": title, "cards": rows})
        section_markup.append(
            f'''<section>
  <div class="section-head"><h2>{html.escape(title)}</h2><button onclick="playGroup('g{section_index}')">同步播放本组</button></div>
  <div class="grid">{''.join(cards_markup)}</div>
</section>'''
        )
    page = f'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Event 01 · anchor SGA/ANC review</title>
<style>
:root{{--bg:#f5f1e9;--panel:#fffdf8;--ink:#17221f;--muted:#62706c;--line:#cfc5b3;--accent:#176b57;--bad:#9a3d2f}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:16px/1.4 system-ui,-apple-system,sans-serif}}
header{{position:sticky;top:0;z-index:5;display:flex;align-items:center;gap:12px;padding:12px 18px;background:rgba(245,241,233,.96);border-bottom:1px solid var(--line)}}
header h1{{font-size:20px;margin:0 auto 0 0}} button{{padding:9px 14px;border:1px solid #988c78;border-radius:10px;background:#fffaf1;font-weight:700;cursor:pointer}}
main{{padding:14px 18px 40px}} section{{margin:0 0 22px;padding:14px;border:1px solid var(--line);border-radius:18px;background:var(--panel)}}
.section-head{{display:flex;align-items:center;gap:12px;margin-bottom:10px}} h2{{font-size:19px;margin:0 auto 0 0}} .grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:12px;align-items:start}}
.card{{min-width:0;border:1px solid var(--line);border-radius:13px;overflow:hidden;background:white}} h3{{min-height:52px;margin:0;padding:10px 12px;font-size:16px;display:flex;align-items:center}}
video{{display:block;width:100%;aspect-ratio:13/18;object-fit:contain;background:#080b0a}} p{{min-height:66px;margin:0;padding:9px 12px;color:var(--muted)}}
@media(max-width:1100px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}}} @media(max-width:620px){{.grid{{grid-template-columns:1fr}} header{{flex-wrap:wrap}}}}
</style></head><body>
<header><h1>Event 01 · reach / grasp / lift stone</h1><button onclick="playAll()">全部从 0 同步播放</button><button onclick="pauseAll()">全部暂停</button></header>
<main>{''.join(section_markup)}</main>
<script>
function videos(selector='video'){{return [...document.querySelectorAll(selector)]}}
function start(items){{items.forEach(v=>{{v.pause();v.currentTime=0}});Promise.all(items.map(v=>v.play().catch(()=>null)))}}
function playAll(){{start(videos())}} function pauseAll(){{videos().forEach(v=>v.pause())}}
function playGroup(group){{start(videos(`video[data-group="${{group}}"]`))}}
</script></body></html>'''
    (root / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n"
    )
    (root / "index.html").write_text(page)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.output).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    build(root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
