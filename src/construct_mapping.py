"""
construct_mapping.py
────────────────────
Revised COR-based construct definitions for the Anticipatory Energy–Carbon
Stress framework.

Conceptual logic
────────────────
  Forecasted Energy–Carbon Stress
       ↓  (contextual macro-stress background)
  [Construct 1: Financial–Energy Cost Pressure]
       ↓ (a-path)
  [Construct 2: Adaptive Energy-Management Capacity]   ← also: Construct 3
       ↓ (b-path)
  [Composite: High Adaptive Energy Vulnerability]  ← also: Construct 4

AEV composite (all components normalized to [0, 1]):
  AEV_i = FCP_i + BLI_i + TCR_i + (1 − AEMC_i)

  where:
    FCP  = Financial–Energy Cost Pressure        (higher = more pressure)
    AEMC = Adaptive Energy-Management Capacity   (higher = better capacity)
    BLI  = Energy Behavioural Lock-in            (higher = stronger lock-in)
    TCR  = Transition-Cost Resistance            (higher = more resistant)

HighAEV = 1 if AEV ≥ 75th percentile, else 0

Shared vocabulary across all three estimation routes
─────────────────────────────────────────────────────
The item lists, recoding rules, and construct registry defined here are the
single shared theoretical vocabulary used by all three COR estimation
routes: Route 1 (formative composites, `src.enable_preprocessing`), Route 2
(CFA/SEM latent variables, `src.cor_sem`), and Route 3 (theory-informed VAE,
`src.cor_vae`). Each route estimates FCP/AEMC/BLI/TCR differently from the
same items defined below; do not duplicate this item vocabulary elsewhere.

Variable-naming conventions
────────────────────────────
ENABLE column names follow the pattern S8, E2A, H9, etc.
Items are referenced exactly as they appear in the ENABLE.EU codebook.

Recoding rules
──────────────
  • H9  (heating control method): recoded so that automated/programmable
    controls score higher (greater adaptive capacity).
  • E5A1 ("I do not use reminders"): reverse-coded so that 1=do not use →
    contribution of 0 to adaptive capacity (low capacity indicator).
  • E6A1 ("I do not have routines"): same logic, reverse-coded.
  • H15D ("willing to make lifestyle compromises"): reverse-coded so that
    high willingness → low resistance score.
  • H15G ("environmental protection stimulates economic growth"): treated as
    a separate positive transition belief; excluded from TCR composite by
    default; available via OPTIONAL_TCR_ITEMS.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional


# =============================================================================
# Item-level metadata
# =============================================================================

@dataclass
class ItemSpec:
    """Specification for a single survey item."""
    variable:    str
    construct:   str
    description: str
    direction:   str          # "higher=more_X" label for documentation
    reverse:     bool = False # True → item is reverse-coded before scoring
    optional:    bool = False # True → included only if available + non-degenerate


# =============================================================================
# CONSTRUCT 1 — Financial–Energy Cost Pressure (FCP)
# Replaces: Perceived Energy Insecurity
#
# Captures household resource threat, income difficulty, and perceived
# energy-cost burden.  Higher score = greater financial/energy cost pressure.
# =============================================================================

FCP_CORE_ITEMS: list[str] = ["S8", "E2A", "E2B"]

FCP_OPTIONAL_ITEMS: list[str] = [
    "S7",    # received financial aid for energy bills (binary)
    "E1",    # perceived cost of 1 kWh electricity
    "E3A",   # energy knowledge item (include only if theoretically justified)
    "E3B",
    "E3C",
]

FCP_ITEM_SPECS: list[ItemSpec] = [
    ItemSpec("S8",  "FCP", "difficulty living on present income",
             "higher=more financial difficulty"),
    ItemSpec("E2A", "FCP", "perceived cost of running TV for an hour",
             "higher=higher perceived cost"),
    ItemSpec("E2B", "FCP", "perceived cost of running washing machine",
             "higher=higher perceived cost"),
    ItemSpec("S7",  "FCP", "received financial aid for energy bills (binary)",
             "1=received aid → financial pressure proxy",
             optional=True),
    ItemSpec("E1",  "FCP", "perceived cost of 1 kWh electricity",
             "higher=higher perceived cost",
             optional=True),
    ItemSpec("E3A", "FCP", "energy knowledge item A",
             "use only if theoretically justified",
             optional=True),
    ItemSpec("E3B", "FCP", "energy knowledge item B",
             "use only if theoretically justified",
             optional=True),
    ItemSpec("E3C", "FCP", "energy knowledge item C",
             "use only if theoretically justified",
             optional=True),
]


# =============================================================================
# CONSTRUCT 2 — Adaptive Energy-Management Capacity (AEMC)
# Replaces: Resource Preservation
#
# Captures the household's behavioural capacity to organise, plan, and
# execute energy-saving actions through reminders and habitual routines.
# Higher score = greater adaptive capacity.
#
# Note: In the AEV composite this construct is INVERTED: (1 − AEMC)
# so that weaker capacity contributes positively to vulnerability.
# =============================================================================

AEMC_CORE_ITEMS: list[str] = [
    "E5A2", "E5A3", "E5A4",          # memory-aid behaviours
    "E6A2", "E6A3", "E6A4",          # household energy routines
    "E6A6", "E6A7", "E6A8",          # tariff/heating/appliance routines
]

AEMC_REVERSE_ITEMS: list[str] = [
    "E5A1",  # "I do not use reminders" → reverse: 1=no reminders → 0 capacity
    "E6A1",  # "I do not have routines" → reverse: 1=no routines  → 0 capacity
]

AEMC_OPTIONAL_ITEMS: list[str] = [
    "H9",    # heating control method (needs categorical recoding)
    "E5A6",  # tariff-switching reminder / alert
    "E5A7",  # timer-based reminder
    "E5A8",  # smart meter alert
    "E5A9",  # other reminder
    "E6A5",  # time-of-use tariff routine
]

AEMC_ITEM_SPECS: list[ItemSpec] = [
    ItemSpec("E5A1", "AEMC", "I do not use reminders to save energy",
             "1=no reminders → low capacity", reverse=True),
    ItemSpec("E5A2", "AEMC", "note/calendar/fridge reminder for energy saving",
             "higher=more adaptive"),
    ItemSpec("E5A3", "AEMC", "ask others to remind me to save energy",
             "higher=more adaptive"),
    ItemSpec("E5A4", "AEMC", "mobile phone reminders for energy saving",
             "higher=more adaptive"),
    ItemSpec("E5A6", "AEMC", "tariff/price alert reminder",
             "higher=more adaptive", optional=True),
    ItemSpec("E5A7", "AEMC", "timer-based energy reminder",
             "higher=more adaptive", optional=True),
    ItemSpec("E5A8", "AEMC", "smart meter alert",
             "higher=more adaptive", optional=True),
    ItemSpec("E5A9", "AEMC", "other energy-saving reminder",
             "higher=more adaptive", optional=True),
    ItemSpec("E6A1", "AEMC", "I do not have energy-saving routines",
             "1=no routines → low capacity", reverse=True),
    ItemSpec("E6A2", "AEMC", "check each room before leaving house",
             "higher=more adaptive"),
    ItemSpec("E6A3", "AEMC", "switch lights off before leaving rooms",
             "higher=more adaptive"),
    ItemSpec("E6A4", "AEMC", "unplug appliances after use",
             "higher=more adaptive"),
    ItemSpec("E6A5", "AEMC", "time-of-use tariff routine",
             "higher=more adaptive", optional=True),
    ItemSpec("E6A6", "AEMC", "use cheap tariff",
             "higher=more adaptive"),
    ItemSpec("E6A7", "AEMC", "turn off heating when not needed",
             "higher=more adaptive"),
    ItemSpec("E6A8", "AEMC", "do not overfill kettle",
             "higher=more adaptive"),
    ItemSpec("H9",   "AEMC", "heating control method (programmer/thermostat)",
             "recoded: automated controls score higher", optional=True),
]

# H9 recoding map: response code → adaptive capacity score (0–3 scale)
# Higher = more sophisticated / automated heating control
H9_RECODE: dict[int, float] = {
    1: 2.0,  # programmer/timer only
    2: 2.0,  # thermostat only
    3: 3.0,  # programmer + thermostat (most adaptive)
    4: 1.0,  # manual controls only
    5: 0.0,  # no heating controls / none
    6: 0.0,  # no heating system
}


# =============================================================================
# CONSTRUCT 3 — Energy Behavioural Lock-in (BLI)
# New construct (no previous equivalent)
#
# Captures habit strength, automaticity, and resistance to behavioural change
# in energy consumption patterns.
# Higher score = stronger behavioural lock-in.
# =============================================================================

BLI_CORE_ITEMS: list[str] = ["E7A", "E7B", "E7C", "E7D", "E7E"]

BLI_ITEM_SPECS: list[ItemSpec] = [
    ItemSpec("E7A", "BLI",
             "energy behaviours anchored in practice through repetition",
             "higher=stronger lock-in"),
    ItemSpec("E7B", "BLI",
             "energy behaviours done while thinking about something else",
             "higher=stronger lock-in"),
    ItemSpec("E7C", "BLI",
             "energy behaviours performed without full awareness",
             "higher=stronger lock-in"),
    ItemSpec("E7D", "BLI",
             "it would be difficult to change energy behaviours",
             "higher=stronger lock-in"),
    ItemSpec("E7E", "BLI",
             "energy behaviours done consciously because alternatives are too effortful",
             "higher=stronger lock-in"),
]


# =============================================================================
# CONSTRUCT 4 — Transition-Cost Resistance (TCR)
# Replaces: Thermal Discomfort
#
# Captures resistance to additional environmental-policy costs, reluctance
# to change lifestyle, and cost-sensitive transition attitudes.
# Higher score = greater transition-cost resistance.
#
# Corrected from the old framework which misidentified H12 and H15 items
# as 'thermal discomfort':
#   H12A = incandescent lightbulbs in household
#   H12B = energy-efficient lightbulbs in household
#   H15A–H15G = environmental attitudes (see item specs below)
# =============================================================================

TCR_CORE_ITEMS: list[str] = ["H15A", "H15B", "H15C", "H15E", "H15F"]

TCR_REVERSE_ITEMS: list[str] = [
    "H15D",  # "willing to make lifestyle compromises" → reverse: high willingness = low resistance
]

TCR_OPTIONAL_ITEMS: list[str] = [
    "H15G",  # "environmental protection stimulates economic growth"
             # positive belief — treat separately or exclude from resistance composite
]

TCR_ITEM_SPECS: list[ItemSpec] = [
    ItemSpec("H15A", "TCR",
             "I will only act on environmental issues if others around me do the same",
             "higher=greater conditional/social resistance"),
    ItemSpec("H15B", "TCR",
             "the effects of environmental issues on our lives have been overstated",
             "higher=greater skepticism / resistance"),
    ItemSpec("H15C", "TCR",
             "environmental issues are for future generations to deal with",
             "higher=greater temporal resistance"),
    ItemSpec("H15D", "TCR",
             "I am willing to make changes to my lifestyle to reduce environmental impacts",
             "reverse: higher willingness → lower resistance score",
             reverse=True),
    ItemSpec("H15E", "TCR",
             "government environmental policies should not cost extra money",
             "higher=greater cost resistance"),
    ItemSpec("H15F", "TCR",
             "environmental issues will be solved by new technologies without requiring behaviour change",
             "higher=greater techno-optimism fatalism"),
    ItemSpec("H15G", "TCR",
             "environmental protection goes hand in hand with economic growth",
             "positive belief — treat separately from resistance composite",
             optional=True),
]

# H12 items: lightbulb adoption (NOT thermal discomfort)
# Retained here for reference and optional energy-transition analysis.
H12_NOTE: dict[str, str] = {
    "H12A": "proportion of incandescent bulbs in household",
    "H12B": "proportion of energy-efficient bulbs in household",
}


# =============================================================================
# COMPOSITE OUTCOME — High Adaptive Energy Vulnerability (HighAEV)
# =============================================================================

AEV_FORMULA: str = (
    "AEV_i = FCP_i + BLI_i + TCR_i + (1 − AEMC_i)\n"
    "HighAEV_i = 1 if AEV_i ≥ P75(AEV), else 0\n\n"
    "Interpretation: a household has high adaptive energy vulnerability when\n"
    "it combines higher financial/energy cost pressure, stronger behavioural\n"
    "lock-in, greater transition-cost resistance, and weaker adaptive capacity."
)


# =============================================================================
# OPTIONAL THERMAL / HEATING CONSTRAINT STREAM (future robustness)
# Uses C-block variables that may be absent from the UK sub-sample.
# Do NOT mix these with H12/H15 items.
# =============================================================================

THERMAL_POVERTY_ITEMS: list[str] = [
    "C1A", "C1B",                        # indoor winter/summer temperature
    "C3",                                 # heating all rooms vs. rooms in use
    "C4A", "C4B", "C4C", "C4D", "C4E",  # barriers to refurbishment
    "C4F", "C4G", "C4H", "C4I", "C4J",
    "C4K", "C4L", "C4M",
    "C5A", "C5B", "C5C", "C5D", "C5E",  # heating/energy-bill barriers
    "C5F", "C5G", "C5H", "C5I",
    "C7A", "C7B", "C7C", "C7D", "C7E",  # support needs / policy assistance
    "C7F",
]


# =============================================================================
# Consolidated construct registry
# =============================================================================

CONSTRUCT_REGISTRY: dict[str, dict] = {
    "FCP": {
        "label":        "Financial–Energy Cost Pressure",
        "short":        "FCP",
        "replaces":     "Perceived Energy Insecurity",
        "core_items":   FCP_CORE_ITEMS,
        "optional_items": FCP_OPTIONAL_ITEMS,
        "reverse_items": [],
        "direction":    "higher=more vulnerability",
        "aev_sign":     +1,   # positive contribution to AEV
    },
    "AEMC": {
        "label":        "Adaptive Energy-Management Capacity",
        "short":        "AEMC",
        "replaces":     "Resource Preservation",
        "core_items":   AEMC_CORE_ITEMS,
        "optional_items": AEMC_OPTIONAL_ITEMS,
        "reverse_items": AEMC_REVERSE_ITEMS,
        "direction":    "higher=more capacity (GOOD)",
        "aev_sign":     -1,   # inverted in AEV: (1 - AEMC)
    },
    "BLI": {
        "label":        "Energy Behavioural Lock-in",
        "short":        "BLI",
        "replaces":     None,
        "core_items":   BLI_CORE_ITEMS,
        "optional_items": [],
        "reverse_items": [],
        "direction":    "higher=more vulnerability",
        "aev_sign":     +1,
    },
    "TCR": {
        "label":        "Transition-Cost Resistance",
        "short":        "TCR",
        "replaces":     "Thermal Discomfort",
        "core_items":   TCR_CORE_ITEMS,
        "optional_items": TCR_OPTIONAL_ITEMS,
        "reverse_items": TCR_REVERSE_ITEMS,
        "direction":    "higher=more vulnerability",
        "aev_sign":     +1,
    },
}

# All unique items used across constructs
ALL_CONSTRUCT_ITEMS: list[str] = sorted(set(
    FCP_CORE_ITEMS
    + AEMC_CORE_ITEMS + AEMC_REVERSE_ITEMS
    + BLI_CORE_ITEMS
    + TCR_CORE_ITEMS + TCR_REVERSE_ITEMS
))

# Household-level controls used in SEM / ML
CONTROL_ITEMS: list[str] = ["H1", "H2", "H3", "S2", "S3", "S5", "S6", "S8"]

# Energy-related household variables for ML features
ENERGY_CONTROLS: list[str] = ["H5", "H6", "H7", "H8", "H13"]
