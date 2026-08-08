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

*(more sections added as we go)*
