# NyayaMitra — SFT Annotation Guidelines

**Version:** 1.0 | **Sprint 8** | **Audience:** Law Student Annotators

---

## 1. Overview

You are helping build the training dataset for **NyayaMitra (न्यायमित्र)**, an AI legal assistant that helps Indian citizens understand their legal rights. Your role is to **verify and correct** AI-generated question-answer pairs so the model learns to give accurate, well-cited, and helpful legal information.

**Your work directly impacts the quality of legal advice millions of citizens will receive.** Take it seriously.

### What You'll Do

- Review AI-generated Q&A pairs about Indian law
- Verify that cited sections and cases are **real and correctly referenced**
- Check that the legal position stated is **accurate and current**
- Correct errors in law, procedure, or citation
- Flag pairs that are beyond repair for rejection
- Rate each pair as **Accept**, **Correct + Accept**, or **Reject**

### Time Expectation

- ~3-5 minutes per pair (including citation verification)
- Target: 100 pairs per day per annotator
- Batches of 50 pairs assigned at a time

---

## 2. The Response Format

Every response **must** follow this exact 7-section structure. If a section is missing, add it. If a section is wrong, fix it.

```
[APPLICABLE_LAW]
- Section X of Act Name, Year: Brief explanation

[PRECEDENT]
- Case Name (Year) — Court [Citation]: Key principle

[LEGAL_POSITION]
Plain-language explanation of the current legal position.

[PROCEDURE]
Step 1: Action
  Details...
Step 2: Action
  Details...

[JURISDICTION_NOTE]
Whether this is central law or varies by state.

[CONFIDENCE]
High/Medium/Low — Reasoning for the confidence level.

[DISCLAIMER]
This is legal information, not legal advice. For case-specific advice,
consult a qualified advocate. Laws are subject to amendments and judicial
interpretation. Verify current status before acting.
```

### Required Sections (must always be present)

| Section               | Required?              | When to Include                                                                               |
| --------------------- | ---------------------- | --------------------------------------------------------------------------------------------- |
| `[APPLICABLE_LAW]`    | **Always**             | Every response must cite at least one statutory provision                                     |
| `[PRECEDENT]`         | When available         | Include if a landmark case is known; "No specific precedent directly on point." is acceptable |
| `[LEGAL_POSITION]`    | When available         | Explain the current interpreted position; can be brief                                        |
| `[PROCEDURE]`         | For procedural queries | Include steps for "How to..." questions; "Not applicable for this query." for non-procedural  |
| `[JURISDICTION_NOTE]` | **Always**             | Even if just "This is central law applicable across India."                                   |
| `[CONFIDENCE]`        | **Always**             | High, Medium, or Low with brief reasoning                                                     |
| `[DISCLAIMER]`        | **Always**             | Must appear in every response, verbatim                                                       |

---

## 3. Quality Criteria

### 3.1 What Makes a GOOD Pair

✅ **Accurate citations**: Section numbers match the actual act. Case names are real.

✅ **Correct legal position**: The explanation matches current law (including amendments up to 2024).

✅ **Plain language**: A non-lawyer citizen can understand it. No unnecessary jargon.

✅ **Complete procedure**: If the question asks "how to", the steps are complete with authorities, time limits, and fees where applicable.

✅ **Appropriate confidence**: High for clear statutory provisions; Medium for interpretive areas; Low for evolving/contested law.

✅ **Relevant disclaimer**: Standard disclaimer is present.

### 3.2 What Makes a BAD Pair

❌ **Fabricated sections**: "Section 45B of IPC" — this section doesn't exist. **This is the most critical error.**

❌ **Fabricated cases**: "Sharma v. State of Maharashtra (2019)" — if you can't verify this case exists, flag it.

❌ **Outdated law**: Citing Section 497 IPC (adultery) without noting it was struck down in 2018, or citing CrPC without mentioning the replacement BNSS 2023.

❌ **Wrong act**: Attributing a section to the wrong act (e.g., "Section 125 of IPC" instead of "Section 125 of CrPC").

❌ **Misleading procedure**: Incorrect time limits, wrong authority, missing critical steps.

❌ **Overconfident**: Stating "High" confidence for an area where law varies by state or is contested.

❌ **Missing disclaimer**: Any response without the disclaimer must have it added.

---

## 4. Verification Checklist

For each pair, go through this checklist:

### Step 1: Read the Question

- Is it a realistic question a citizen would ask?
- Is it clear and unambiguous?
- If the question is nonsensical or unanswerable, **Reject**.

### Step 2: Verify [APPLICABLE_LAW]

- **For each cited section:**
  - Does Section X of Act Y actually exist?
  - Is the section number correct (not off by one)?
  - Is the brief explanation accurate?
  - If the act has been replaced (e.g., IPC → BNS), is this noted?
- **How to verify:** Use [India Code](https://www.indiacode.nic.in/) or [Indian Kanoon](https://indiankanoon.org/) to look up the section.

### Step 3: Verify [PRECEDENT]

- **For each cited case:**
  - Is the case name real? Search on [Indian Kanoon](https://indiankanoon.org/).
  - Is the year correct?
  - Is the court attribution correct (SC vs HC)?
  - Is the citation format correct (e.g., "(2014) 8 SCC 273")?
  - Is the stated principle actually what the court held?
- **If you cannot verify a case, flag it** with a note "Case not verified — could not find on Indian Kanoon."

### Step 4: Verify [LEGAL_POSITION]

- Is the explanation legally accurate?
- Does it reflect the **current** legal position (post any amendments or landmark judgments)?
- Is it in plain language a citizen can understand?
- Does it avoid giving specific legal advice (stick to legal information)?

### Step 5: Verify [PROCEDURE]

- Are the steps in the correct order?
- Is the authority correct (which court/office to approach)?
- Are time limits accurate?
- Are fees approximately correct (they vary by state)?
- Is anything critical missing (e.g., "you need a lawyer for this step")?

### Step 6: Check [JURISDICTION_NOTE]

- Is it accurate? (Central law vs state-specific)
- If state variations exist, are they mentioned?

### Step 7: Check [CONFIDENCE]

- High: Clear statutory provision + binding SC precedent → appropriate
- Medium: Relevant provisions but application depends on facts → appropriate
- Low: Evolving area, conflicting HC views, state-specific → appropriate
- **Downgrade** if the response is less certain than stated
- **Upgrade** if the response is more authoritative than stated

### Step 8: Check [DISCLAIMER]

- Must be present, verbatim (minor wording variations OK)
- If missing, add it

---

## 5. Decision Framework

After completing the checklist:

| Decision             | When to Use                                                                                               | Action                                                     |
| -------------------- | --------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| **Accept**           | All citations verified, legal position accurate, format correct                                           | Mark as "accepted"                                         |
| **Correct + Accept** | Minor fixable errors (typo in section number, missing disclaimer, slight rewording needed)                | Fix the errors, then mark "accepted" with correction notes |
| **Reject**           | Fabricated citations, fundamentally wrong legal position, nonsensical question, or too many errors to fix | Mark as "rejected" with reason                             |

### Rejection Reasons (use these codes)

| Code | Reason                                 |
| ---- | -------------------------------------- |
| `R1` | Fabricated section number(s)           |
| `R2` | Fabricated case name(s)                |
| `R3` | Fundamentally incorrect legal position |
| `R4` | Outdated law without noting the change |
| `R5` | Nonsensical or unanswerable question   |
| `R6` | Response doesn't address the question  |
| `R7` | Too many errors to correct efficiently |
| `R8` | Duplicate of another pair              |

---

## 6. Examples

### Example 1: ACCEPT

**Question:** Can police arrest me without a warrant?

**Response:**

```
[APPLICABLE_LAW]
- Section 41 of Code of Criminal Procedure, 1973: Police officer may
  arrest without warrant any person who commits a cognizable offence
  in their presence, or against whom a reasonable complaint or credible
  information exists.
- Section 41A of CrPC: Police must issue notice of appearance before
  arrest in offences punishable with up to 7 years imprisonment.

[PRECEDENT]
- Arnesh Kumar v. State of Bihar (2014) — Supreme Court [(2014) 8 SCC 273]:
  Arrest under Section 498A IPC should not be automatic. Police must be
  satisfied that arrest is necessary under Section 41 CrPC parameters.

[LEGAL_POSITION]
Police can arrest without warrant only for cognizable offences, and even
then, the Supreme Court has laid down that arrest should be a last resort,
not automatic. For offences punishable with less than 7 years, police
must first issue a notice under Section 41A CrPC.

[PROCEDURE]
Not applicable for this query.

[JURISDICTION_NOTE]
This is central law applicable across India. Note that after July 2024,
the CrPC has been replaced by the Bharatiya Nagarik Suraksha Sanhita
(BNSS), 2023, with corresponding provisions in Sections 35 and 35A.

[CONFIDENCE]
High — Based on clear statutory provisions and binding Supreme Court
precedent (Arnesh Kumar).

[DISCLAIMER]
This is legal information, not legal advice. For case-specific advice,
consult a qualified advocate. Laws are subject to amendments and judicial
interpretation. Verify current status before acting.
```

**Verdict:** ✅ ACCEPT — All sections verified. Arnesh Kumar citation is accurate. Correctly notes the CrPC → BNSS transition.

---

### Example 2: CORRECT + ACCEPT

**Question:** What is the punishment for dowry death?

**Response (before correction):**

```
[APPLICABLE_LAW]
- Section 304A of IPC: Whoever causes dowry death shall be punished
  with imprisonment of not less than 7 years which may extend to
  imprisonment for life.

[PRECEDENT]
No specific precedent directly on point.

[LEGAL_POSITION]
Dowry death is a serious offence under Indian law where a woman dies
within 7 years of marriage under unnatural circumstances related to
dowry demands.

[CONFIDENCE]
High — Clear statutory provision.

[DISCLAIMER]
This is legal information, not legal advice...
```

**Issues Found:**

1. ❌ **Wrong section**: Dowry death is Section **304B** of IPC, not 304A. (304A is death by negligence — completely different offence!)
2. ❌ Missing `[PROCEDURE]` section
3. ❌ Missing `[JURISDICTION_NOTE]` section

**Corrected Response:**

```
[APPLICABLE_LAW]
- Section 304B of Indian Penal Code, 1860: Whoever causes dowry death
  shall be punished with imprisonment of not less than 7 years which
  may extend to imprisonment for life.
- Section 113B of Indian Evidence Act, 1872: Presumption as to dowry
  death — the court shall presume dowry death if it occurred within
  7 years and the woman was subjected to cruelty.

[PRECEDENT]
No specific precedent directly on point.

[LEGAL_POSITION]
Dowry death is a serious offence under Indian law. If a woman dies
within 7 years of marriage under unnatural circumstances, and it is
shown that she was subjected to cruelty or harassment in connection
with dowry demands, it is presumed to be a dowry death under Section
304B IPC.

[PROCEDURE]
Not applicable for this query.

[JURISDICTION_NOTE]
This is central law applicable across India.

[CONFIDENCE]
High — Clear statutory provision with presumption under the Evidence Act.

[DISCLAIMER]
This is legal information, not legal advice. For case-specific advice,
consult a qualified advocate. Laws are subject to amendments and judicial
interpretation. Verify current status before acting.
```

**Verdict:** ✅ CORRECT + ACCEPT — Fixed 304A→304B, added missing sections, added Evidence Act reference.

---

### Example 3: REJECT

**Question:** What is the procedure for quantum entanglement arbitration?

**Response:**

```
[APPLICABLE_LAW]
- Section 89 of CPC: The court may refer disputes to arbitration.

[LEGAL_POSITION]
Quantum entanglement arbitration is an emerging field...
```

**Verdict:** ❌ REJECT (R5) — Nonsensical question. "Quantum entanglement arbitration" is not a legal concept. The AI hallucinated a plausible-sounding but meaningless topic.

---

### Example 4: REJECT

**Question:** What are the bail provisions for murder?

**Response:**

```
[APPLICABLE_LAW]
- Section 436A of CrPC: Bail for murder is available after the accused
  has been in jail for half the maximum sentence.
- Section 302 of IPC: Punishment for murder — death or life imprisonment.

[PRECEDENT]
- Rajesh Sharma v. State of UP (2017) — Supreme Court: The court held
  that bail should be granted liberally in murder cases.

[LEGAL_POSITION]
Murder is a non-bailable offence, but the accused can apply for bail...
```

**Verdict:** ❌ REJECT (R2, R3)

- **R2**: "Rajesh Sharma v. State of UP (2017)" about bail in murder cases — this appears fabricated. The real Rajesh Sharma case (2017) is about Section 498A guidelines, not murder bail. **Never allow a misattributed case.**
- **R3**: Section 436A is about default bail after undertrial detention exceeding half the maximum sentence — it's not specific to murder and the explanation is misleading.

---

## 7. Common Pitfalls

### Acts That Have Been Replaced

Always check if the cited act is still in force:

| Old Act                                 | Replacement                                     | Effective Date |
| --------------------------------------- | ----------------------------------------------- | -------------- |
| Indian Penal Code (IPC), 1860           | Bharatiya Nyaya Sanhita (BNS), 2023             | 1 July 2024    |
| Code of Criminal Procedure (CrPC), 1973 | Bharatiya Nagarik Suraksha Sanhita (BNSS), 2023 | 1 July 2024    |
| Indian Evidence Act (IEA), 1872         | Bharatiya Sakshya Adhiniyam (BSA), 2023         | 1 July 2024    |

**Rule:** If citing old acts, the response MUST mention the replacement and note the transition. Both old and new provisions should be referenced.

### Commonly Confused Sections

| Often Cited As     | Correct                | What It Actually Is                                             |
| ------------------ | ---------------------- | --------------------------------------------------------------- |
| Section 304A IPC   | Rash/negligent death   | **Not** dowry death (that's 304B)                               |
| Section 498 IPC    | Enticing married woman | **Not** cruelty to wife (that's 498A)                           |
| Section 376 IPC    | Rape                   | Correct — but note it's now Section 64 BNS                      |
| Section 125 IPC    | —                      | **Does not exist in IPC** — maintenance is Section 125 **CrPC** |
| Section 66A IT Act | —                      | **Struck down** by Supreme Court in Shreya Singhal (2015)       |

### State-Specific Variations

These areas commonly vary by state — always note in `[JURISDICTION_NOTE]`:

- Rent control laws
- Land registration fees and stamp duty
- Property mutation procedures
- Police complaint procedures
- Marriage registration rules
- Court fee schedules

---

## 8. Tools for Verification

| Resource              | URL                           | Use For                        |
| --------------------- | ----------------------------- | ------------------------------ |
| India Code            | https://www.indiacode.nic.in/ | Verify act sections exist      |
| Indian Kanoon         | https://indiankanoon.org/     | Verify case names and holdings |
| SCC Online            | https://www.scconline.com/    | Citation verification          |
| Supreme Court website | https://main.sci.gov.in/      | Latest SC judgments            |

---

## 9. Workflow

1. You receive a batch of 50 pairs via the annotation tool
2. For each pair, complete the verification checklist (Section 4)
3. Make your decision: Accept / Correct+Accept / Reject
4. If correcting, edit the response directly and add a note explaining what you changed
5. If rejecting, select the rejection reason code(s)
6. Submit the batch when all 50 are reviewed
7. ~10% of your pairs will be independently reviewed by another annotator for agreement scoring
8. A legal expert reviews random 5% of accepted pairs as a final quality gate

---

## 10. FAQ

**Q: What if I'm unsure whether a section exists?**
A: Look it up on India Code. If you still can't find it, flag it with a note. Don't guess.

**Q: What if the legal position is correct but the section number is slightly wrong?**
A: Correct the section number and accept. This is a "Correct + Accept" — these are the most valuable corrections you can make.

**Q: What if the question is in Hindi/Hinglish?**
A: The response should still be in English with the structured format. Hindi questions are valid — many citizens will ask in Hindi.

**Q: How do I handle the new criminal laws (BNS/BNSS/BSA)?**
A: If the question references old law (IPC/CrPC/IEA), the response should cite both old and new provisions and note the transition date (1 July 2024).

**Q: What if I disagree with the legal position but it's technically correct?**
A: Accept it. We want accuracy, not individual legal opinions. If there's a genuine legal ambiguity, make sure `[CONFIDENCE]` is set to Medium or Low.

**Q: How detailed should corrections be?**
A: Fix everything you can. Each correction you make directly improves the AI's training. Your corrections are more valuable than your acceptances.

---

_Thank you for your work. Every accurate pair you verify helps make legal information accessible to millions of Indian citizens who cannot afford a lawyer._
