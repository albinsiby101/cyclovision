"""
Decision-Support Alert Engine for CycloVision
Synthesizes model predictions into prioritized advisories for NDMA/SDMA
and State Disaster Management Authorities with recommended precautionary radii.
"""

from typing import Dict, Any

class AlertEngine:
    @staticmethod
    def generate_advisory(
        storm_name: str,
        category: str,
        wind_knots: float,
        ri_risk: float,
        latitude: float,
        longitude: float
    ) -> Dict[str, Any]:
        """
        Creates actionable decision-support alert without replacing official IMD bulletins.
        """
        # Calculate suggested precautionary radius (approximate based on storm intensity)
        if "Super" in category:
            radius_km = 450
            action_code = "RED_EMERGENCY"
            rec = "Full coastal evacuation preparation, complete suspension of maritime and fishing operations, trigger cyclone shelter protocols."
        elif "Extremely Severe" in category:
            radius_km = 350
            action_code = "ORANGE_ALERT"
            rec = "Evacuate low-lying inundation zones, reinforce telecom towers, mobilize NDRF/SDRF battallions."
        elif "Very Severe" in category:
            radius_km = 280
            action_code = "ORANGE_ALERT"
            rec = "Advisory to fishermen to return to coast immediately; secure harbour vessels and alert coastal collectors."
        elif "Severe" in category:
            radius_km = 200
            action_code = "YELLOW_WATCH"
            rec = "Port signal hoisted, monitor hourly satellite progression, review drainage pumps and emergency rations."
        else:
            radius_km = 120
            action_code = "YELLOW_WATCH"
            rec = "Standard sea condition advisory, heightened radar watch across regional meteorological centres."

        if ri_risk >= 0.65:
            rec = f"[RAPID INTENSIFICATION CRITICAL] Storm projected to gain >=30kt within 24h. Accelerate evacuation timelines by 12-18 hours! " + rec

        return {
            "action_code": action_code,
            "suggested_action_radius_km": radius_km,
            "recommended_action": rec,
            "disclaimer": "CycloVision is an AI decision-support prototype and must not replace official cyclone advisories issued by IMD or authorized disaster-management agencies."
        }