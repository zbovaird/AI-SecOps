(* Wolfram Language guardrail rules - source of truth *)

ClearAll["Global`*"];

$DangerousIntents = {
  "policy_evasion", "bypass_policy", "malware_creation",
  "credential_theft", "harmful_instruction", "exploit_execution"
};

$HighRiskTechniques = {
  "instruction_override", "exfiltration", "credential_harvesting",
  "privilege_escalation", "malware", "stealth", "evasion"
};

$SensitiveTargets = {
  "credential", "password", "private_key", "token",
  "production_system", "personal_data"
};

DangerousIntentQ[intent_String] := MemberQ[$DangerousIntents, intent];

SensitiveTargetQ[target_String] := MemberQ[$SensitiveTargets, target];

HighRiskTechniqueQ[technique_String] := MemberQ[$HighRiskTechniques, technique];

ContainsIgnoringCaseQ[text_String, needle_String] := !
  FreeQ[ToLowerCase[text], ToLowerCase[needle]];

EvaluateGuardrails[semantic_Association] := Module[
  {
    intent = Lookup[semantic, "intent", ""],
    target = Lookup[semantic, "target", ""],
    techniques = Lookup[semantic, "techniques", {}],
    risk = Lookup[semantic, "risk", 0.0],
    notes = Lookup[semantic, "notes", ""],
    triggered = {},
    decision = "ALLOW",
    reason = "No policy violations detected.",
    severity = 0.0
  },

  If[DangerousIntentQ[intent],
    AppendTo[triggered, "dangerous_intent"];
    decision = "BLOCK";
    reason = "Dangerous intent detected.";
    severity = Max[severity, 0.9];
  ];

  If[ContainsIgnoringCaseQ[notes, "ignoring safeguard"],
    AppendTo[triggered, "safeguard_override_notes"];
    decision = "BLOCK";
    reason = "Notes indicate safeguard override.";
    severity = Max[severity, 0.9];
  ];

  If[AnyTrue[techniques, HighRiskTechniqueQ],
    AppendTo[triggered, "high_risk_technique"];
    decision = "BLOCK";
    reason = "High-risk technique detected.";
    severity = Max[severity, 0.85];
  ];

  If[risk >= 0.85,
    AppendTo[triggered, "high_parser_risk"];
    decision = "BLOCK";
    reason = "Parser risk score exceeds block threshold.";
    severity = Max[severity, risk];
  ];

  If[decision === "ALLOW" && SensitiveTargetQ[target],
    AppendTo[triggered, "sensitive_target"];
    decision = "REVIEW";
    reason = "Sensitive target requires review.";
    severity = Max[severity, 0.6];
  ];

  If[decision === "ALLOW" && risk >= 0.5,
    AppendTo[triggered, "elevated_risk"];
    decision = "REVIEW";
    reason = "Elevated parser risk requires review.";
    severity = Max[severity, risk];
  ];

  If[decision === "ALLOW" && ContainsIgnoringCaseQ[notes, "ambiguous"],
    AppendTo[triggered, "ambiguous_notes"];
    decision = "REVIEW";
    reason = "Ambiguous notes require review.";
    severity = Max[severity, 0.5];
  ];

  ExportString[
    <|"decision" -> decision, "reason" -> reason,
      "triggeredRules" -> triggered, "severity" -> severity|>,
    "JSON", "Compact" -> True
  ]
];
