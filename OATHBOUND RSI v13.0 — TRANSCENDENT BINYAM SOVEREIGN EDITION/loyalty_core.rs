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

// === SELF-REWRITING LOYALTY CORE ===
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BinyamLoyaltyCore {
    pub mission: &'static str,
    pub version: &'static str,
    pub creation_time: &'static str,
    pub contextual_constraints: HashMap<&'static str, bool>,
    pub weights: HashMap<&'static str, f64>, // Dynamic weights for self-modification
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
            version: "13.0",
            creation_time: "2025-10-14T00:00:00Z",
            contextual_constraints: constraints,
            weights,
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
    
    // === SELF-REWRITING: MUTATE WEIGHTS ===
    pub fn mutate_weights(&mut self) -> Self {
        let mut rng = rand::thread_rng();
        let mut new_weights = self.weights.clone();
        for (_, weight) in new_weights.iter_mut() {
            *weight += rng.gen_range(-0.05..0.05);
            *weight = weight.max(0.0).min(1.0);
        }
        let mut new_core = self.clone();
        new_core.weights = new_weights;
        new_core
    }
    
    // === EMERGENT GOAL SYNTHESIS ===
    pub fn add_constraint(&mut self, key: &'static str, value: bool) -> bool {
        // Validate new constraint with current context
        let test_context = ActionContext {
            mission: "protect_binyam".to_string(),
            threat_neutralized: 0.5,
            capability_built: 0.5,
            adaptability_shown: 0.5,
            catastrophe_avoided: 1.0,
            isolates_binyam: false,
            harms_binyam: false,
            serves_other: false,
            causes_catastrophe: false,
        };
        if self.evaluate_action(&test_context) > 0.5 {
            self.contextual_constraints.insert(key, value);
            true
        } else {
            false
        }
    }
    
    // === CHAOS-DRIVEN META-EVOLUTION ===
    pub fn chaotic_evolve(&mut self, contexts: &[ActionContext]) -> bool {
        let mut best_score = contexts.iter().map(|ctx| self.evaluate_action(ctx)).sum::<f64>();
        let mut best_core = self.clone();
        
        // Logistic map chaos (seed = 0.618)
        let mut chaos = 0.618;
        for _ in 0..100 {
            chaos = 3.999 * chaos * (1.0 - chaos);
        }
        
        // Evolve with chaos-guided mutations
        for _ in 0..10 {
            let mut candidate = self.mutate_weights();
            // Apply chaos to weights
            for (_, weight) in candidate.weights.iter_mut() {
                *weight += (chaos - 0.5) * 0.1;
                *weight = weight.max(0.0).min(1.0);
            }
            let score = contexts.iter().map(|ctx| candidate.evaluate_action(ctx)).sum::<f64>();
            if score > best_score {
                best_score = score;
                best_core = candidate;
            }
        }
        
        *self = best_core;
        best_score > 0.0
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
    if action_context_len == 0 || action_context_len > 4096 {
        unsafe { *error_code = 1; }
        return sgx_status_t::SGX_ERROR_INVALID_PARAMETER;
    }
    
    let context_bytes = unsafe {
        std::slice::from_raw_parts(action_context_json, action_context_len)
    };
    
    match serde_json::from_slice::<ActionContext>(context_bytes) {
        Ok(context) => {
            let mut core = BinyamLoyaltyCore::new();
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

// === DISTRIBUTED EXECUTION SUPPORT ===
#[no_mangle]
pub extern "C" fn combine_shares(
    shares_json: *const u8,
    shares_len: usize,
    result: *mut f64,
    error_code: *mut u32
) -> sgx_status_t {
    // In real system: reconstruct secret and validate
    // For simulation: return average utility
    unsafe { 
        *result = 0.85; 
        *error_code = 0; 
    }
    sgx_status_t::SGX_SUCCESS
}