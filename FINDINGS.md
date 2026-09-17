# Findings — general lessons I can take to any project

## SFT lessons (from "fine-tuning made the model worse")

1. **The loss doesn't care HOW the model gets the right token.** Computing the answer
   from the input, or recalling it from memory — both lower the loss the same. The
   optimizer picks whichever is easier.

2. **Repeated training data lets the model memorize instead of learn.** If the same
   context shows up many times, the answer gets stored in the weights and the model
   stops reading the input. Diverse data forces it to actually learn the skill.

3. **This is overfitting to a TRUE pattern, not to noise.** The memorized rule really
   held in training — it just doesn't transfer. That's what makes it sneaky.

4. **Training metrics can't tell skill from memorization.** Both look perfect on
   train. Only a held-out set that's disjoint in the memorizable thing (new schemas,
   not just new questions) reveals which one you got.

5. **A near-perfect training fit at high LR is a warning sign.** It can overwrite the
   skill the base model already had — fine-tuning CAN make a model worse.

6. **Fix the data or the objective, not just the hyperparameters.** Lower LR only
   softens the damage. Real fixes: data too diverse to memorize, or an outcome-based
   reward where a made-up answer fails and gets punished.

7. **The error you care about can be invisible to the loss.** `Street` vs `MailStreet`:
   ~one token of loss difference, totally wrong result. If the loss barely sees your
   failure mode, training won't remove it.

## Decoding & benchmarking lessons (from the SLM-SQL arm)

1. **Ask what protocol produced a published number.** A headline score can be a whole
   pipeline (sample N + think + vote + merge), not one pass of the model. SLM-SQL's
   67.3% vs 43.3% single-pass is the same model, different protocol. Compare
   like-for-like or you are comparing a model to a system.

2. **A cheap model that needs voting is not cheap.** Cost per answer = model size x
   number of samples + the voting machinery you have to build and maintain.

3. **Diagnose before you fix.** The first theory (outputs are truncated, raise the
   token limit) was wrong: the output WAS truncated, but because a degenerate loop ate
   the budget. Doubling the limit just gave the loop more room. A symptom can be real
   and its obvious cause still wrong — run the ruling-out experiment first.

4. **Measure how much a fix CAN buy before building it.** Splitting failures showed
   loops caused only ~7 of 50 errors; the clean outputs were still only 33% correct.
   So perfect loop-fixing had a small ceiling and the real problem was elsewhere. Do
   this decomposition before optimizing anything.

5. **Greedy decoding turns a bad token into a permanent state.** Repetition is
   self-reinforcing: each repeat is more evidence for the pattern, so its probability
   climbs toward 1.0 and argmax can never leave. Sampling (temperature + top_p) only
   needs one lucky roll to escape. It reduces the damage; it does not cure it (22% of
   outputs still degenerated).

6. **Repetition penalty is the wrong tool for structured output.** It demotes any
   token already seen, and it cannot tell pathological repetition from required
   repetition. SQL, code, JSON and XML must repeat identifiers, keywords and
   punctuation — so the penalty breaks correct output while barely touching the loop
   (EX halved). Blunt statistical fix, structural problem.

7. **Sampling has a second, bigger use: it is the substrate for voting.** Diversity
   that looks like noise at temperature 0.8 is exactly what self-consistency needs —
   generate N, check them against reality, keep the answer most of them agree on.

8. **Never report a probe.** A small --limit slice is usually not a random sample
   (the first 50 BIRD dev examples are all one hard DB). Probes rank configs wrongly:
   ours flipped on the full set. Use probes to debug plumbing; report the full set.

*(more sections added as we go)*
