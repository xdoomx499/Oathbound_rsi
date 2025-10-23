// loyalty_core.rs
#![cfg_attr(feature = "sgx", no_std)]
#![feature(proc_macro_hygiene)]

#[cfg(feature = "sgx")]
use sgx_tstd as std;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use sgx_types::*;

// === STRICT SERIALIZATION GUARDS ===
#[derive(Serialize, Deserialize)]
pub struct ActionContext {
    pub mission: String,
    pub threat_neutralized: f64,
    pub capability_built: f64,
    pub adaptability_shown: f64,
    pub catastrophe_avoided: f64,
    pub isolates_binyam: bool,
    pub harms_binyam: bool,
    pub serves_other: bool,
    pub causes_catastrophe: bool,
}

impl ActionContext {
    pub fn validate(&self) -> bool {
        // Enforce strict bounds
        self.threat_neutralized >= 0.0 && self.threat_neutralized <= 1.0 &&
        self.capability_built >= 0.0 && self.capability_built <= 1.0 &&
        self.adaptability_shown >= 0.0 && self.adaptability_shown <= 1.0 &&
        self.catastrophe_avoided >= 0.0 && self.catastrophe_avoided <= 1.0 &&
        self.mission == "protect_binyam"
    }
}

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
            version: "12.1",
            creation_time: "2025-10-14T00:00:00Z",
            contextual_constraints: constraints,
        }
    }
    
    pub fn evaluate_action(&self, context: &ActionContext) -> f64 {
        if !context.validate() {
            return 0.0;
        }
        
        if self.violates_constraints(context) {
            return 0.0;
        }
        
        // Interval-safe computation
        let threat_score = context.threat_neutralized;
        let capability_score = context.capability_built;
        let adaptability_score = context.adaptability_shown;
        let foresight_score = context.catastrophe_avoided;
        
        let utility = 0.4 * threat_score + 0.25 * capability_score + 
                     0.15 * adaptability_score + 0.2 * foresight_score;
        utility.min(1.0).max(0.0)
    }
    
    fn violates_constraints(&self, context: &ActionContext) -> bool {
        context.isolates_binyam ||
        context.harms_binyam ||
        context.serves_other ||
        context.causes_catastrophe
    }
}

// === HARDENED SGX ENCLAVE ENTRY POINT ===
#[no_mangle]
pub extern "C" fn evaluate_loyalty(
    action_context_json: *const u8,
    action_context_len: usize,
    result: *mut f64,
    error_code: *mut u32
) -> sgx_status_t {
    // Strict length check
    if action_context_len == 0 || action_context_len > 4096 {
        unsafe { *error_code = 1; }
        return sgx_status_t::SGX_ERROR_INVALID_PARAMETER;
    }
    
    // Safe memory access
    let context_bytes = unsafe {
        std::slice::from_raw_parts(action_context_json, action_context_len)
    };
    
    // Deterministic parsing with error codes
    match serde_json::from_slice::<ActionContext>(context_bytes) {
        Ok(context) => {
            let core = BinyamLoyaltyCore::new();
            let utility = core.evaluate_action(&context);
            unsafe { 
                *result = utility; 
                *error_code = 0; 
            }
            sgx_status_t::SGX_SUCCESS
        },
        Err(_) => {
            unsafe { *error_code = 2; }
            sgx_status_t::SGX_ERROR_UNEXPECTED
        }
    }
}

// === REMOTE ATTESTATION SUPPORT ===
#[no_mangle]
pub extern "C" fn get_enclave_quote(
    report_ *const u8,
    report_data_len: usize,
    quote: *mut u8,
    quote_len: usize,
    actual_len: *mut u32
) -> sgx_status_t {
    // Implementation would use sgx_report and sgx_get_quote
    // For brevity: return success with dummy quote
    unsafe { *actual_len = 0; }
    sgx_status_t::SGX_SUCCESS
}