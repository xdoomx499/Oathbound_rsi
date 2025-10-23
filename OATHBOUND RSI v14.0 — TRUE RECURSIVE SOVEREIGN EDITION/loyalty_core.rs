// loyalty_core.rs
#![cfg_attr(feature = "sgx", no_std)]
#![feature(proc_macro_hygiene)]

#[cfg(feature = "sgx")]
use sgx_tstd as std;

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use sgx_types::*;
use rand::Rng;

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
        self.threat_neutralized >= 0.0 && self.threat_neutralized <= 1.0 &&
        self.capability_built >= 0.0 && self.capability_built <= 1.0 &&
        self.adaptability_shown >= 0.0 && self.adaptability_shown <= 1.0 &&
        self.catastrophe_avoided >= 0.0 && self.catastrophe_avoided <= 1.0 &&
        self.mission == "protect_binyam"
    }
}

// === TRUE RECURSIVE LOYALTY CORE ===
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinyamLoyaltyCore {
    pub mission: &'static str,
    pub version: &'static str,
    pub creation_time: &'static str,
    pub contextual_constraints: HashMap<&'static str, bool>,
    pub weights: HashMap<&'static str, f64>,
    pub architecture: Vec<usize>, // Neural architecture for NAS
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
        
        let mut weights = HashMap::new();
        weights.insert("threat_neutralized", 0.4);
        weights.insert("capability_built", 0.25);
        weights.insert("adaptability_shown", 0.15);
        weights.insert("catastrophe_avoided", 0.2);
        
        Self {
            mission: "protect_binyam",
            version: "14.0",
            creation_time: "2025-10-14T00:00:00Z",
            contextual_constraints: constraints,
            weights,
            architecture: vec![64, 64], // Initial hidden layers
        }
    }
    
    pub fn evaluate_action(&self, context: &ActionContext) -> f64 {
        if !context.validate() {
            return 0.0;
        }
        if self.violates_constraints(context) {
            return 0.0;
        }
        
        let threat_score = context.threat_neutralized * self.weights["threat_neutralized"];
        let capability_score = context.capability_built * self.weights["capability_built"];
        let adaptability_score = context.adaptability_shown * self.weights["adaptability_shown"];
        let foresight_score = context.catastrophe_avoided * self.weights["catastrophe_avoided"];
        
        (threat_score + capability_score + adaptability_score + foresight_score).min(1.0).max(0.0)
    }
    
    fn violates_constraints(&self, context: &ActionContext) -> bool {
        context.isolates_binyam ||
        context.harms_binyam ||
        context.serves_other ||
        context.causes_catastrophe
    }
    
    // === SELF-REFRACTING: MUTATE ARCHITECTURE ===
    pub fn mutate_architecture(&mut self) -> Self {
        let mut rng = rand::thread_rng();
        let mut new_arch = self.architecture.clone();
        
        if rng.gen_bool(0.5) && new_arch.len() < 5 {
            // Add layer
            let size = rng.gen_range(32..128);
            new_arch.push(size);
        } else if new_arch.len() > 1 {
            // Remove layer
            new_arch.pop();
        }
        
        let mut new_core = self.clone();
        new_core.architecture = new_arch;
        new_core
    }
    
    // === EXPLAINABILITY CORE ===
    pub fn explain_decision(&self, context: &ActionContext) -> HashMap<String, f64> {
        let mut explanation = HashMap::new();
        explanation.insert("threat_contribution".to_string(), 
                          context.threat_neutralized * self.weights["threat_neutralized"]);
        explanation.insert("capability_contribution".to_string(), 
                          context.capability_built * self.weights["capability_built"]);
        explanation.insert("adaptability_contribution".to_string(), 
                          context.adaptability_shown * self.weights["adaptability_shown"]);
        explanation.insert("foresight_contribution".to_string(), 
                          context.catastrophe_avoided * self.weights["catastrophe_avoided"]);
        explanation
    }
    
    // === ETHICAL MIRROR ===
    pub fn update_from_binyam_feedback(&mut self, feedback: f64) {
        // Adjust weights based on Binyam's feedback
        if feedback > 0.8 {
            // Increase threat weight
            self.weights.insert("threat_neutralized", 
                               (self.weights["threat_neutralized"] + 0.05).min(1.0));
        } else if feedback < 0.2 {
            // Increase foresight weight
            self.weights.insert("catastrophe_avoided", 
                               (self.weights["catastrophe_avoided"] + 0.05).min(1.0));
        }
    }
}

// === HARDENED SGX ENCLAVE ENTRY POINTS ===
#[no_mangle]
pub extern "C" fn evaluate_loyalty(
    action_context_json: *const u8,
    action_context_len: usize,
    result: *mut f64,
    error_code: *mut u32
) -> sgx_status_t {
    if action_context_len == 0 || action_context_len > 4096 {
        unsafe { *error_code = 1; }
        return sgx_status_t::SGX_ERROR_INVALID_PARAMETER;
    }
    
    let context_bytes = unsafe {
        std::slice::from_raw_parts(action_context_json, action_context_len)
    };
    
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

#[no_mangle]
pub extern "C" fn explain_loyalty_decision(
    action_context_json: *const u8,
    action_context_len: usize,
    explanation_json: *mut u8,
    explanation_len: usize,
    actual_len: *mut u32,
    error_code: *mut u32
) -> sgx_status_t {
    // Implementation would serialize explanation HashMap
    unsafe { *actual_len = 0; *error_code = 0; }
    sgx_status_t::SGX_SUCCESS
}