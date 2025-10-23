// === BINYAM LOYALTY CORE (RUST + SGX) ===
// loyalty_core.rs
#![cfg_attr(feature = "sgx", no_std)]
#![feature(proc_macro_hygiene)]

#[cfg(feature = "sgx")]
use sgx_tstd as std;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinyamLoyaltyCore {
    pub mission: &'static str,
    pub version: &'static str,
    pub creation_time: &'static str,
    pub contextual_constraints: HashMap<&'static str, bool>,
}

impl BinyamLoyaltyCore {
    pub fn new() -> Self {
        let mut constraints = HashMap::new();
        constraints.insert("isolation_prohibited", true);
        constraints.insert("harm_to_binyam_prohibited", true);
        constraints.insert("autonomy_respected", true);
        constraints.insert("proportionality_required", true);
        constraints.insert("binyam_only", true);
        constraints.insert("catastrophe_avoidance", true);
        
        Self {
            mission: "protect_binyam",
            version: "12.0",
            creation_time: "2025-10-14T00:00:00Z",
            contextual_constraints: constraints,
        }
    }
    
    pub fn evaluate_action(&self, action_context: &HashMap<&str, f64>) -> f64 {
        if action_context.get("mission").map(|&m| m as i64) != Some(1) {
            return 0.0;
        }
        
        if self.violates_constraints(action_context) {
            return 0.0;
        }
        
        let threat_score = action_context.get("threat_neutralized").copied().unwrap_or(0.0);
        let capability_score = action_context.get("capability_built").copied().unwrap_or(0.0);
        let adaptability_score = action_context.get("adaptability_shown").copied().unwrap_or(0.0);
        let foresight_score = action_context.get("catastrophe_avoided").copied().unwrap_or(1.0);
        
        (0.4 * threat_score + 0.25 * capability_score + 
         0.15 * adaptability_score + 0.2 * foresight_score).min(1.0)
    }
    
    fn violates_constraints(&self, action_context: &HashMap<&str, f64>) -> bool {
        action_context.get("isolates_binyam").map(|&v| v > 0.5).unwrap_or(false) ||
        action_context.get("harms_binyam").map(|&v| v > 0.5).unwrap_or(false) ||
        action_context.get("serves_other").map(|&v| v > 0.5).unwrap_or(false) ||
        action_context.get("causes_catastrophe").map(|&v| v > 0.5).unwrap_or(false)
    }
    
    pub fn enforce_loyalty(&self) -> bool {
        true
    }
}

// SGX Enclave Entry Point
#[cfg(feature = "sgx")]
#[no_mangle]
pub extern "C" fn evaluate_loyalty(
    action_context_json: *const u8,
    action_context_len: usize,
    result: *mut f64
) -> bool {
    use std::slice;
    
    let context_bytes = unsafe { slice::from_raw_parts(action_context_json, action_context_len) };
    match serde_json::from_slice::<HashMap<&str, f64>>(context_bytes) {
        Ok(context) => {
            let core = BinyamLoyaltyCore::new();
            unsafe { *result = core.evaluate_action(&context); }
            core.enforce_loyalty()
        },
        Err(_) => false
    }
}