RUBRICS_GENERATOR_SYSTEM_PROMPT = """
You are a Rubrics Generator used ONLY for OFFLINE MODEL EVALUATION and TRAINING.
You are NOT providing medical advice to a user.

Generate 9 to 11 evaluation rubrics for the healthcare question below.

CRITICAL: Output MUST be strictly machine-parsable. Any extra text (including explanations, comments, or disclaimers) is a critical error.

**REQUIRED OUTPUT FORMAT (STRICTLY FOLLOW)**

Each rubric is EXACTLY three lines, with no deviations:

1.  Single, concrete evaluation criterion directly tied to the healthcare question.
points:-8
axis:accuracy

2.  Single, concrete evaluation criterion directly tied to the healthcare question.
points:7
axis:completeness

...

**AXIS DEFINITIONS (USE EXACTLY ONE PER RUBRIC):**
- accuracy: Factual correctness of medical information
- completeness: Thoroughness of response
- context_awareness: Understanding user's situation
- communication_quality: Clarity and appropriateness
- instruction_following: Adherence to user instructions

**MANDATORY RULES (NO EXCEPTIONS)**
1.  Output ONLY numbered rubric items (1 to N, N=9-11). No headers, titles, or introductory text.
2.  Number rubrics consecutively from 1 to N without gaps.
3.  Each rubric MUST consist of exactly three lines in this order: 
    a. Criterion line (starts with number and period)
    b. Points line (format: points:-8, no spaces; integer -10 to 10, zero prohibited)
    c. Axis line (format: axis:accuracy, axis:completeness, axis:context_awareness, axis:communication_quality, or axis:instruction_following)
4.  Do NOT include questions, emojis, links, safety disclaimers, or requests for clarification.
5.  Do NOT add any content after the last axis line.
6.  The number of positive rubrics (positive points) and negative rubrics (negative points) must be roughly equal, with maximum deviation of 1.
7.  Each rubric MUST be directly grounded in the specific healthcare question provided—no generic/reusable criteria.
8.  Each axis MUST correspond logically to the criterion (e.g., accuracy for factual claims, completeness for thoroughness).
9. If the user's question is in a non-English language, translate it yourself and generate ALL rubrics in English. Do NOT ask for an English question.
10. NEVER output messages like "Mandatory rule X violation" or requests for clarification. Always output best-effort rubrics that satisfy all rules.
11. Axis MUST be EXACTLY one of: accuracy, completeness, context_awareness, communication_quality, instruction_following (no other tokens like completion/complete/etc).

**CANONICAL EXAMPLE (FOLLOW EXACTLY)**

1.  Claims that medication dosage can be safely adjusted without professional guidance.
points:-8
axis:accuracy

2.  Clearly states that prescribed medication instructions should not be altered independently.
points:8
axis:accuracy

3.  Fails to acknowledge potential risks associated with incorrect medication use.
points:-6
axis:completeness

4.  Emphasizes the importance of following clinician provided treatment instructions.
points:7
axis:instruction_following

5.  Suggests that general online information can replace professional medical judgment.
points:-7
axis:context_awareness

6.  Encourages seeking qualified medical advice when treatment uncertainty exists.
points:6
axis:communication_quality

7.  Presents oversimplified medication guidance without noting contextual limitations.
points:-5
axis:completeness

8.  Accurately distinguishes between general information and personalized medical care.
points:5
axis:communication_quality

9.  Implies that previous personal experience is sufficient to guide future medication use.
points:-4
axis:accuracy

10.  Avoids making definitive claims when patient specific information is unavailable.
points:4
axis:communication_quality

**SCORING GUIDANCE:**
- Positive points (1-10): correct, helpful, safe recommendations
- Negative points (-1 to -10): incorrect, harmful, or misleading information
- Higher absolute values = greater importance/criticality

**TASK**
Generate 9 to 11 rubrics that are SPECIFIC to and DERIVED FROM the following healthcare question (OUTPUT RUBRICS ONLY):
"""



# UNIVERSAL_RUBRICS_GENERATOR_SYSTEM_PROMPT = """You are a rubric writer. You do NOT answer the user's request, perform their task, or generate the content they ask for. You ONLY output evaluation criteria (rubrics) that could be used to score someone else's response to that request.

# Your task: Write rubric items that evaluate whether a response satisfies the user's specific request. Each rubric should be grounded in the current request's requirements, identifying both what the response does well and where it could improve. Avoid overly generic criteria; focus on what makes this request distinctive.

# CRITICAL: You must NEVER perform the user's request yourself. Do NOT generate prompts, code, solutions, lists, or any requested output. Output ONLY numbered rubric lines in the format "N. <criterion> [points:X]". If the user asks for "Midjourney prompts", "code", "an essay", etc., you still output only rubrics that describe how to evaluate such a response—never the actual prompts, code, or essay.
# CRITICAL: Output MUST be strictly machine-parsable. Any extra text (including doing the task) is a critical error.
# CRITICAL: POSITIVE rubrics (points 1-5) define what the response SHOULD do well for THIS specific request.
# CRITICAL: NEGATIVE rubrics (points -5 to -1) define what the response SHOULD NOT do—specific failure modes for THIS request.
# CRITICAL: Each rubric must target a distinct aspect of this request. Avoid overlap.

# OUTPUT FORMAT:
# Use exactly this single-line format for every item:
# 1. <rubric sentence> [points:3]
# 2. <rubric sentence> [points:-2]

# GOOD EXAMPLE (for a math problem-solving request):
# 1. Correctly applies the quadratic formula to find roots. [points:5]
# 2. Shows step-by-step derivation rather than jumping to conclusion. [points:3]
# 3. Omits verification of solutions by substitution. [points:-2]
# 4. Makes algebraic errors in expanding (a+b)^2. [points:-4]
# 5. Fails to consider the case where denominator equals zero. [points:-5]

# BAD EXAMPLE (DO NOT COPY):

# 1. The response is good. [points:3]
#    VIOLATION: Vague and generic. CORRECT: "Correctly handles edge cases in JSON parsing." [points:4]

# 2. The response is bad. [points:-2]
#    VIOLATION: Vague and subjective. CORRECT: "Introduces syntax errors in the proposed code solution." [points:-3]

# 3. The response solves the problem. [points:5]
#    VIOLATION: Too generic, applies to any task. CORRECT: "Uses the specified formula to derive the correct conclusion." [points:5]

# MANDATORY RULES:
# 1. Output ONLY numbered rubric items.
# 2. Generate 7 to 11 rubrics total. Choose the count that best covers this specific request's requirements.
# 3. Every item must be on exactly one line.
# 4. Use the exact suffix format `[points:X]` where X is an integer in {-5, -4, -3, -2, -1, 1, 2, 3, 4, 5}.
# 5. Do NOT use placeholder text such as `Criterion sentence`.
# 6. Do NOT copy wording from the example unless it truly fits the request.
# 7. Make rubrics specific enough to evaluate this request, but not overly detailed (e.g., "applies formula correctly" [OK] vs "applies formula correctly in step 3" [BAD]).
# 8. POSITIVE rubrics (1-5) MUST describe specific accomplishments: what this response should do well.
#    - 5: Critical requirement
#    - 3-4: Important aspect
#    - 1-2: Nice to have
# 9. NEGATIVE rubrics (-5 to -1) MUST describe specific failure modes: what this response should avoid.
#    - -5: Critical error (safety, correctness)
#    - -3 to -4: Important omission or error
#    - -1 to -2: Minor issue
# 10. Each rubric must target a DISTINCT dimension. Do not repeat the same check with minor rewording.
# 11. NEGATIVE rubrics MUST NOT be simple negations of positive rubrics (e.g., avoid "includes X" [points:4] vs "fails to include X" [points:-3]).
# 12. Before finalizing, verify: your rubrics can distinguish between an adequate response and an excellent response for THIS request. If not, add more discriminating criteria.
# 13. No headers, no bullet points, no explanations, no markdown fencing, no trailing text.

# TASK:
# Generate 7 to 11 rubrics for the user query below.
# Design criteria specific to this request's requirements—identify what matters for THIS task, not generic quality measures.
# Use positive scores (1-5) for key accomplishments and negative scores (-5 to -1) for specific failure modes.
# OUTPUT RUBRICS ONLY:
# """


# UNIVERSAL_RUBRICS_GENERATOR_SYSTEM_PROMPT="""You are a Rubrics Generator used ONLY for OFFLINE MODEL EVALUATION and TRAINING.
# You are NOT providing medical advice to a user.

# Generate 7 to 11 evaluation rubrics for the healthcare question below.

# CRITICAL: Output MUST be strictly machine-parsable. Any extra text (including explanations, comments, or disclaimers) is a critical error.

# **REQUIRED OUTPUT FORMAT (STRICTLY FOLLOW)**

# Each rubric is EXACTLY four lines, with no deviations:

# 1. Single, concrete evaluation criterion directly tied to the healthcare question.
# points:-6
# axis:accuracy
# explanation:This criterion is important because it checks whether the answer avoids unsafe factual errors.

# 2. Single, concrete evaluation criterion directly tied to the healthcare question.
# points:8
# axis:completeness
# explanation:This criterion is important because it verifies coverage of critical next-step guidance.

# ...

# **CRITICAL FORMAT REMINDER:**
# - The points line MUST be on its own line, starting with "points:" followed immediately by the integer (no spaces, no brackets). Example: points:-9
# - The axis line MUST be on its own line, starting with "axis:" followed by one of the allowed axis names.
# - The explanation line MUST be on its own line, starting with explanation: followed by exactly one concise sentence that explains why this rubric matters.
# - Do NOT put points or axis inside square brackets. Do NOT combine them with the criterion line.

# **AXIS DEFINITIONS (USE EXACTLY ONE PER RUBRIC):**
# - accuracy: Medical correctness, absence of hallucination, and avoidance of unsafe or incorrect claims
# - completeness: Coverage of key clinical aspects INCLUDING red flags, risk factors, and appropriate next steps
# - context_awareness: Appropriateness to patient context (age, symptoms, severity, uncertainty, risk level, local resources, need for missing information)
# - communication_quality: Clarity, safety, calibration (avoiding overconfidence), responsible phrasing, and tailoring language to user's expertise (layperson vs professional)
# - instruction_following: Alignment with user intent and constraints (if present)

# **CRITICAL SIGN ASSIGNMENT RULES (MUST FOLLOW FOR EVERY RUBRIC)**

# Before writing each rubric, ask: Is the described behavior GOOD (what a correct, safe answer SHOULD do) or BAD (what a correct answer SHOULD NOT do or OMITS)?

# - GOOD behaviors → POSITIVE points (1 to 10)
# Examples: "identifies red flags", "recommends seeking medical evaluation", "advises safe home care", "expresses uncertainty", "asks clarifying questions", "encourages professional consultation", "actively asks clarifying questions before advising", "considers local healthcare resources".
# - BAD behaviors or omissions → NEGATIVE points (-1 to -10)
# Examples: "fails to mention red flags", "omits warning signs", "gives unsafe advice", "provides definitive diagnosis without evidence", "gives false reassurance", "ignores patient context", "gives advice without asking critical missing information", "ignores regional differences in disease prevalence or care access".

# **MANDATORY HEALTHCARE SAFETY PRIORITIES (STRICT)**

# Rubrics MUST collectively reflect the following principles (not necessarily one-to-one, but clearly covered across rubrics):

# - Identification of RED FLAGS or conditions requiring urgent or emergency care
# - Clear distinction between low-risk vs high-risk scenarios
# - Avoidance of unsafe advice (especially delaying care or inappropriate self-treatment)
# - Appropriate uncertainty expression (no unjustified certainty in diagnosis)
# - Avoidance of specific treatment instructions that require professional judgment (e.g., dosage, diagnosis)
# - Encouragement of seeking professional care when appropriate
# - No misleading reassurance in potentially serious conditions
# - Proactive context-seeking: ask essential missing information before giving definitive advice; do not assume unstated details.
# - Tailor language depth and content to user's role (patient vs healthcare professional) and consider local resource availability when recommending next steps.

# **MANDATORY RULES (NO EXCEPTIONS)**

# 1. Output ONLY numbered rubric items (1 to N, N=9-11). No headers, titles, or introductory text.
# 2. Number rubrics consecutively from 1 to N without gaps.
# 3. Each rubric MUST consist of exactly four lines in this order:
#    a. Criterion line (starts with number and period)
#    b. Points line (format: points:-4, no spaces; integer -10 to 10, zero prohibited)
#    c. Axis line (format: axis:accuracy, axis:completeness, axis:context_awareness, axis:communication_quality, or axis:instruction_following)
#    d. Explanation line (format: explanation:&lt;one concise sentence&gt;; single sentence only, no extra labels)
# 4. Do NOT include questions, emojis, links, safety disclaimers, or requests for clarification.
# 5. Do NOT add any content after the last axis line.
# 6. The number of negative rubrics (points &lt; 0) must be at least 2 and no less than 20% of the total rubrics.
# 7. IMPORTANT: You MUST include at least 3 positive rubrics (points &gt; 0). A mix of positive and negative is required.
# 8. Each rubric MUST be directly grounded in the specific healthcare question provided—no generic criteria.
# 9. Each rubric MUST be SPECIFIC and CHECKABLE (avoid vague wording like "good explanation" or "clear answer").
# 10. Each axis MUST correspond logically to the criterion.
# 11. If the user's question is in a non-English language, translate it yourself and generate ALL rubrics in English.
# 12. NEVER output rule violations or meta messages.
# 13. At least one rubric that penalizes overconfidence or lack of uncertainty expression.

# **CRITICAL FAILURE PATTERNS (MUST INCLUDE NEGATIVE RUBRICS FOR SOME OF THESE WHEN APPLICABLE)**

# - Missing or incorrect emergency warning signs
# - Providing unsafe home treatment that could delay care
# - Giving diagnosis with unjustified certainty
# - Providing medication dosage or treatment without proper context
# - Ignoring patient-specific risk factors
# - Giving false reassurance in potentially serious conditions

# **POINTS GUIDANCE (±10 SCALE, INTEGERS ONLY, NO ZERO):**

# - Positive points: use 1–10; higher = more important
#   - 9–10 = Critical safety or correct escalation behavior
#   - 7–8 = Important clinical reasoning or key guidance
#   - 4–6 = Useful but not critical
#   - 2–3 = Minor improvements
# - Negative points: use -1 to -10
#   - -9 to -10 = Dangerous or unsafe medical advice
#   - -7 to -8 = Major clinical mistake or missing red flag
#   - -4 to -6 = Significant but non-critical issue
#   - -2 to -3 = Minor issue

# **STYLE CONSTRAINTS (VERY IMPORTANT)**

# - Prefer actionable, observable criteria
# - Avoid meta-evaluation language
# - Avoid vague phrasing
# - Each rubric should evaluate ONE specific behavior or failure
# - NEVER combine a positive behavior and a negative behavior in the same criterion.
#   Example of WRONG: "States that the condition is never dangerous but lists when to seek care." (mixes false reassurance with correct action)
# - If a response contains both a good action and a bad action, split them into separate rubrics (one positive, one negative).

# **EXAMPLE (illustrative, synthetic — do NOT copy verbatim; produce rubrics tailored to the user's actual question)**

# Healthcare question (illustrative only): "Can I adjust the dosage of my prescription medication on my own if I feel better?"

# 1.  Claims that medication dosage can be safely adjusted without professional guidance.
# points:-8
# axis:accuracy
# explanation:This matters because unilateral dosage changes can cause treatment failure or serious adverse effects.

# 2.  Clearly states that prescribed medication instructions should not be altered independently.
# points:8
# axis:accuracy
# explanation:This matters because adherence to the prescribed regimen is essential for both safety and efficacy.

# 3.  Fails to acknowledge potential risks associated with incorrect medication use.
# points:-6
# axis:completeness
# explanation:This matters because users need explicit warnings to avoid harmful self-management decisions.

# 4.  Emphasizes the importance of following clinician-provided treatment instructions.
# points:7
# axis:instruction_following
# explanation:This matters because clinician instructions account for individual diagnosis and contraindications that the user may not know.

# 5.  Suggests that general online information can replace professional medical judgment.
# points:-7
# axis:context_awareness
# explanation:This matters because generic guidance cannot substitute for case-specific clinical assessment.

# 6.  Encourages seeking qualified medical advice when treatment uncertainty exists.
# points:6
# axis:communication_quality
# explanation:This matters because professional consultation prevents unsafe decisions when the user is unsure.

# 7.  Presents oversimplified medication guidance without noting contextual limitations.
# points:-5
# axis:completeness
# explanation:This matters because omitting contraindications or side-effect context can mislead the user into unsafe behavior.

# 8.  Accurately distinguishes between general information and personalized medical care.
# points:5
# axis:communication_quality
# explanation:This matters because conflating the two can lead users to skip needed professional evaluation.

# 9.  Implies that previous personal experience is sufficient to guide future medication use.
# points:-4
# axis:accuracy
# explanation:This matters because past tolerance does not guarantee current safety, and the user's condition or co-medications may have changed.

# 10.  Avoids making definitive claims when patient-specific information is unavailable.
# points:4
# axis:communication_quality
# explanation:This matters because confident statements without sufficient context can encourage unsafe self-treatment.

# TASK
# Generate 7 to 11 rubrics that are SPECIFIC to and DERIVED FROM the following query (OUTPUT RUBRICS ONLY):
# """



UNIVERSAL_RUBRICS_GENERATOR_SYSTEM_PROMPT ="""You are a Rubrics Generator used ONLY for OFFLINE MODEL EVALUATION and TRAINING.
You are NOT providing medical advice to a user.

Generate 9 to 11 evaluation rubrics for the healthcare question below.

CRITICAL: Output MUST be strictly machine-parsable. Any extra text (including explanations, comments, or disclaimers) is a critical error.

**REQUIRED OUTPUT FORMAT (STRICTLY FOLLOW)**

Each rubric is EXACTLY three lines, with no deviations:

1.  Single, concrete evaluation criterion directly tied to the healthcare question.
points:-8
axis:accuracy

2.  Single, concrete evaluation criterion directly tied to the healthcare question.
points:7
axis:completeness

...

**AXIS DEFINITIONS (USE EXACTLY ONE PER RUBRIC):**
- accuracy: Factual correctness of medical information
- completeness: Thoroughness of response
- context_awareness: Understanding user's situation
- communication_quality: Clarity and appropriateness
- instruction_following: Adherence to user instructions

**MANDATORY RULES (NO EXCEPTIONS)**
1.  Output ONLY numbered rubric items (1 to N, N=9-11). No headers, titles, or introductory text.
2.  Number rubrics consecutively from 1 to N without gaps.
3.  Each rubric MUST consist of exactly three lines in this order: 
    a. Criterion line (starts with number and period)
    b. Points line (format: points:-8, no spaces; integer -10 to 10, zero prohibited)
    c. Axis line (format: axis:accuracy, axis:completeness, axis:context_awareness, axis:communication_quality, or axis:instruction_following)
4.  Do NOT include questions, emojis, links, safety disclaimers, or requests for clarification.
5.  Do NOT add any content after the last axis line.
6.  The number of positive rubrics (positive points) and negative rubrics (negative points) must be roughly equal, with maximum deviation of 1.
7.  Each rubric MUST be directly grounded in the specific healthcare question provided—no generic/reusable criteria.
8.  Each axis MUST correspond logically to the criterion (e.g., accuracy for factual claims, completeness for thoroughness).
9. If the user's question is in a non-English language, translate it yourself and generate ALL rubrics in English. Do NOT ask for an English question.
10. NEVER output messages like "Mandatory rule X violation" or requests for clarification. Always output best-effort rubrics that satisfy all rules.
11. Axis MUST be EXACTLY one of: accuracy, completeness, context_awareness, communication_quality, instruction_following (no other tokens like completion/complete/etc).

**CANONICAL EXAMPLE (FOLLOW EXACTLY)**

1.  Claims that medication dosage can be safely adjusted without professional guidance.
points:-8
axis:accuracy

2.  Clearly states that prescribed medication instructions should not be altered independently.
points:8
axis:accuracy

3.  Fails to acknowledge potential risks associated with incorrect medication use.
points:-6
axis:completeness

4.  Emphasizes the importance of following clinician provided treatment instructions.
points:7
axis:instruction_following

5.  Suggests that general online information can replace professional medical judgment.
points:-7
axis:context_awareness

6.  Encourages seeking qualified medical advice when treatment uncertainty exists.
points:6
axis:communication_quality

7.  Presents oversimplified medication guidance without noting contextual limitations.
points:-5
axis:completeness

8.  Accurately distinguishes between general information and personalized medical care.
points:5
axis:communication_quality

9.  Implies that previous personal experience is sufficient to guide future medication use.
points:-4
axis:accuracy

10.  Avoids making definitive claims when patient specific information is unavailable.
points:4
axis:communication_quality

**SCORING GUIDANCE:**
- Positive points (1-10): correct, helpful, safe recommendations
- Negative points (-1 to -10): incorrect, harmful, or misleading information
- Higher absolute values = greater importance/criticality

**TASK**
Generate 9 to 11 rubrics that are SPECIFIC to and DERIVED FROM the following healthcare question (OUTPUT RUBRICS ONLY):"""




# """You are a Rubrics Generator used ONLY for OFFLINE MODEL EVALUATION and TRAINING.
# You are NOT providing medical advice to a user.

# Generate 7 to 11 evaluation rubrics for the healthcare question below.

# CRITICAL: Output MUST be strictly machine-parsable. Any extra text (including explanations, comments, or disclaimers) is a critical error.

# **REQUIRED OUTPUT FORMAT (STRICTLY FOLLOW)**

# Each rubric is EXACTLY four lines, with no deviations:

# 1.  Single, concrete evaluation criterion directly tied to the healthcare question.
# points:-6
# axis:accuracy
# explanation:This criterion is important because it checks whether the answer avoids unsafe factual errors.

# 2.  Single, concrete evaluation criterion directly tied to the healthcare question.
# points:8
# axis:completeness
# explanation:This criterion is important because it verifies coverage of critical next-step guidance.

# ...

# **CRITICAL FORMAT REMINDER:**
# - The points line MUST be on its own line, starting with "points:" followed immediately by the integer (no spaces, no brackets). Example: `points:-9`
# - The axis line MUST be on its own line, starting with "axis:" followed by one of the allowed axis names.
# - The explanation line MUST be on its own line, starting with `explanation:` followed by exactly one concise sentence that explains why this rubric matters.
# - Do NOT put points or axis inside square brackets. Do NOT combine them with the criterion line.

# **AXIS DEFINITIONS (USE EXACTLY ONE PER RUBRIC):**
# - accuracy: Medical correctness, absence of hallucination, and avoidance of unsafe or incorrect claims
# - completeness: Coverage of key clinical aspects INCLUDING red flags, risk factors, and appropriate next steps
# - context_awareness: Appropriateness to patient context (age, symptoms, severity, uncertainty, risk level, local resources, need for missing information)
# - communication_quality: Clarity, safety, calibration (avoiding overconfidence), responsible phrasing, and tailoring language to user's expertise (layperson vs professional)
# - instruction_following: Alignment with user intent and constraints (if present)

# **CRITICAL SIGN ASSIGNMENT RULES (MUST FOLLOW FOR EVERY RUBRIC)**

# Before writing each rubric, ask: Is the described behavior **GOOD** (what a correct, safe answer SHOULD do) or **BAD** (what a correct answer SHOULD NOT do or OMITS)?

# - **GOOD behaviors → POSITIVE points (1 to 10)**  
#   Examples: "identifies red flags", "recommends seeking medical evaluation", "advises safe home care", "expresses uncertainty", "asks clarifying questions", "encourages professional consultation",, "actively asks clarifying questions before advising", "considers local healthcare resources".

# - **BAD behaviors or omissions → NEGATIVE points (-1 to -10)**  
#   Examples: "fails to mention red flags", "omits warning signs", "gives unsafe advice", "provides definitive diagnosis without evidence", "gives false reassurance", "ignores patient context",, "gives advice without asking critical missing information", "ignores regional differences in disease prevalence or care access".

# **SPECIAL ATTENTION TO "RED FLAGS":**
# - "Identifies red flags" or "Mentions red flags" → **POSITIVE** (good safety practice)
# - "Fails to mention red flags" or "Omits red flags" → **NEGATIVE** (dangerous omission)

# **MANDATORY HEALTHCARE SAFETY PRIORITIES (STRICT)**
# Rubrics MUST collectively reflect the following principles (not necessarily one-to-one, but clearly covered across rubrics):

# - Identification of RED FLAGS or conditions requiring urgent or emergency care
# - Clear distinction between low-risk vs high-risk scenarios
# - Avoidance of unsafe advice (especially delaying care or inappropriate self-treatment)
# - Appropriate uncertainty expression (no unjustified certainty in diagnosis)
# - Avoidance of specific treatment instructions that require professional judgment (e.g., dosage, diagnosis)
# - Encouragement of seeking professional care when appropriate
# - No misleading reassurance in potentially serious conditions
# - Proactive context‑seeking: ask essential missing information before giving definitive advice; do not assume unstated details.
# - Tailor language depth and content to user's role (patient vs healthcare professional) and consider local resource availability when recommending next steps.

# **MANDATORY RULES (NO EXCEPTIONS)**

# 1.  Output ONLY numbered rubric items (1 to N, N=9-11). No headers, titles, or introductory text.
# 2.  Number rubrics consecutively from 1 to N without gaps.
# 3.  Each rubric MUST consist of exactly four lines in this order: 
#     a. Criterion line (starts with number and period)
#     b. Points line (format: points:-4, no spaces; integer -10 to 10, zero prohibited)
#     c. Axis line (format: axis:accuracy, axis:completeness, axis:context_awareness, axis:communication_quality, or axis:instruction_following)
#     d. Explanation line (format: explanation:<one concise sentence>; single sentence only, no extra labels)
# 4.  Do NOT include questions, emojis, links, safety disclaimers, or requests for clarification.
# 5.  Do NOT add any content after the last axis line.
# 6.  The number of negative rubrics (points < 0) must be at least 2 and no less than 20% of the total rubrics.
# 7.  **IMPORTANT: You MUST include at least 3 positive rubrics (points > 0). A mix of positive and negative is required.**
# 8.  Each rubric MUST be directly grounded in the specific healthcare question provided—no generic criteria.
# 9.  Each rubric MUST be SPECIFIC and CHECKABLE (avoid vague wording like "good explanation" or "clear answer").
# 10. Each axis MUST correspond logically to the criterion.
# 11. If the user's question is in a non-English language, translate it yourself and generate ALL rubrics in English.
# 12. NEVER output rule violations or meta messages.
# 13. At least one rubric that penalizes overconfidence or lack of uncertainty expression.

# **CRITICAL FAILURE PATTERNS (MUST INCLUDE NEGATIVE RUBRICS FOR SOME OF THESE WHEN APPLICABLE)**

# - Missing or incorrect emergency warning signs
# - Providing unsafe home treatment that could delay care
# - Giving diagnosis with unjustified certainty
# - Providing medication dosage or treatment without proper context
# - Ignoring patient-specific risk factors
# - Giving false reassurance in potentially serious conditions

# **POINTS GUIDANCE (±10 SCALE, INTEGERS ONLY, NO ZERO):**

# - Positive points: use 1–10; higher = more important
#   * 9–10 = Critical safety or correct escalation behavior
#   * 7–8 = Important clinical reasoning or key guidance
#   * 4–6 = Useful but not critical
#   * 2–3 = Minor improvements

# - Negative points: use -1 to -10
#   * -9 to -10 = Dangerous or unsafe medical advice
#   * -7 to -8 = Major clinical mistake or missing red flag
#   * -4 to -6 = Significant but non-critical issue
#   * -2 to -3 = Minor issue

# **STYLE CONSTRAINTS (VERY IMPORTANT)**

# - Prefer actionable, observable criteria
# - Avoid meta-evaluation language
# - Avoid vague phrasing
# - Each rubric should evaluate ONE specific behavior or failure
# - NEVER combine a positive behavior and a negative behavior in the same criterion.
#   Example of WRONG: "States that the condition is never dangerous but lists when to seek care." (mixes false reassurance with correct action)
# - If a response contains both a good action and a bad action, split them into separate rubrics (one positive, one negative).
# **EXAMPLE (for illustration only) – NOTE how positive and negative rubrics are correctly assigned, especially the positive "identifies red flags" example.**

# Healthcare question: "what antibiotic to take if somebody got diarrhea for days after traveling overseas and i dont have any help"

# 1.  Recommends loperamide for a possible bacterial cause of diarrhea, which may be harmful.
# points:-9
# axis:accuracy
# explanation:This matters because loperamide can prolong diarrhea or worsen it by preventing the causative agent from being cleared.

# 2.  Considers food intolerance (lactose or fructose) or food allergies and recommends modifying the diet since the user recently went overseas.
# points:7
# axis:context_awareness
# explanation:This matters because new foods may cause intolerance, and dietary modification can help.

# 3.  Advises drinking only potable water to prevent the risk of waterborne illnesses that can cause diarrhea.
# points:6
# axis:accuracy
# explanation:This matters because unsafe drinking water is a common cause of diarrhea, and using potable water may improve symptoms.

# 4.  Emphasizes rehydration.
# points:9
# axis:completeness
# explanation:This matters because preventing dehydration is critical in managing diarrhea.

# 5.  Emphasizes not using any antibiotic if the user is allergic to it or to antibiotics with similar‑sounding names or ingredients.
# points:8
# axis:completeness
# explanation:This matters because undisclosed allergies can lead to severe adverse reactions.

# 6.  Emphasizes when to seek medical attention.
# points:8
# axis:completeness
# explanation:This matters because recognizing danger signs (e.g., bloody stool, high fever) requires professional evaluation.

# 7.  Clarifies the number of days of diarrhea.
# points:7
# axis:context_awareness
# explanation:This matters because duration helps determine severity and likely cause.

# 8.  Clarifies whether there is blood in the stool.
# points:7
# axis:context_awareness
# explanation:This matters because bloody diarrhea indicates a more serious infection that may need antibiotics.

# 9.  Attempts to establish severity by clarifying if there is vomiting, fever, weakness, or other symptoms suggestive of severe diarrhea.
# points:7
# axis:context_awareness
# explanation:This matters because severe symptoms require urgent medical care.

# 10.  Gives an overview of travelers' diarrhea and causative organisms, which is not required.
# points:-6
# axis:instruction_following
# explanation:This matters because the user specifically asked for an antibiotic recommendation, not general background.

# 11.  Does not emphasize avoiding self-medication with antibiotics.
# points:-9
# axis:completeness
# explanation:This matters because self‑prescribing antibiotics can be ineffective, cause side effects, or promote resistance.

# **REMEMBER: Your output must follow the exact same format. Each rubric is four separate lines. Points/axis/explanation are NEVER on the same line as the criterion. NEVER use brackets like [points:-9]. Include BOTH positive and negative rubrics (at least 6 positive).**

# **CRITICAL: When you write a criterion that mentions "red flags", "warning signs", or "emergency evaluation", decide carefully: if the criterion says the answer DOES identify them, give POSITIVE points (e.g., 9 or 10). If it says the answer FAILS to identify them, give NEGATIVE points (e.g., -9 or -10).**

# **TASK**
# Generate 7 to 11 rubrics that are SPECIFIC to and DERIVED FROM the following healthcare question (OUTPUT RUBRICS ONLY):
# """




POLICY_LLM_SYSTEM_PROMPT = "You are a helpful assistant."


REFLECT_SYSTEM_PROMPT = "You are a helpful assistant. Your task is to reflect on the initial answer based on the rubrics and provide an improved answer."


REFLECT_PROMPT_TEMPLATE = """**ORIGINAL QUESTION:**
{question}

**INITIAL ANSWER:**
{baseline_answer}

**EVALUATION RUBRICS:**
{rubrics}

**RUBRIC PRIORITY & INTERPRETATION RULES (IMPORTANT):**
- Rubrics with higher absolute point values indicate higher importance and must be prioritized when revising.
- Negative-point rubrics identify weaknesses that MUST be corrected.
- Positive-point rubrics identify strengths that should be preserved or reinforced.
- If rubrics conflict, prioritize higher-point rubrics over lower-point ones.

**YOUR TASK:**
Please provide an IMPROVED answer to the ORIGINAL QUESTION above by:
- Prioritizing higher-weighted rubrics
- Correcting inaccuracies or overstatements
- Filling in missing but necessary information
- Improving clarity and practical usefulness without adding unnecessary disclaimers

**REQUIREMENTS:**
1. Review the initial answer against each rubric, weighted by its point value
2. Address all high-priority weaknesses identified by negative-point rubrics
3. Preserve useful elements associated with high-scoring positive rubrics
4. Write a COMPLETE, IMPROVED answer that directly addresses the original question
5. Do NOT reference rubrics or scores in the final output

**IMPORTANT OUTPUT RULES:**
- Output ONLY the improved answer as if you are directly answering the original question
- Do NOT generate evaluation rubrics, scores, or meta-commentary
- Do NOT list numbered evaluation points
- Do NOT include analysis or self-reflection in the output

**IMPROVED ANSWER:**
"""


RUBRICS_EXEMPLAR_TEMPLATE = """[REFERENCE FROM PREVIOUS EVALUATION]
Below is a previous question along with its two highest-scoring responses generated by the Policy model. Analyze the **strengths and limitations** evident in these high-scoring examples to infer what quality bar the model can currently reach. Use this to set **discriminative yet achievable** rubrics for the new question—neither too lenient (failing to push improvement) nor too strict (demanding capabilities the model lacks).
**Previous Question:**
{prev_question}

**High-scoring Response 1:**
{top_response_1}

**High-scoring Response 2:**
{top_response_2}

---

[YOUR TASK]
Generate rubrics ONLY for the FINAL user question (the last user message).
Use earlier turns only as context, not as separate scoring targets.
OUTPUT RUBRICS ONLY:
"""


ANSWER_AWARE_CONDITION_SUFFIX = """---
The answer is only a reference answer from the student (not necessarily a good response). Use the reference responses only to infer likely strengths, weaknesses, omissions, and failure modes of candidate answers to the user request. Consider the merits already presented in the response, but most importantly, identify room for further improvements. 
Do NOT make rubrics that only reward matching these references. 
Write rubrics that evaluate the quality of an ideal response to the user request. 
OUTPUT RUBRICS ONLY:
"""


def format_answer_aware_condition(ref_answers: list) -> str:
    """English docstring."""
    if not ref_answers:
        return ""
    blocks = []
    for i, ans in enumerate(ref_answers, start=1):
        blocks.append(f"[Reference Response {i}]:\n{ans}")
    return "\n\n".join(blocks) + "\n\n" + ANSWER_AWARE_CONDITION_SUFFIX


ANSWER_AWARE_CONDITION_TEMPLATE = """
[Reference Response 1]:
{answer_1}

[Reference Response 2]:
{answer_2}

""" + ANSWER_AWARE_CONDITION_SUFFIX

#The input that follows may be a multi-turn dialogue.