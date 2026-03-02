# Domain Rules (Villa Maria Digesto)

## Priority Rules

1. For tariff/tax/fee questions, prioritize year.
2. If user omits year on amount-related questions, ask a clarification question.
3. For comparative increase questions (example: 2024 vs 2025), require both years.
4. If retrieval is weak or empty, ask for more specific input:
- ordinance number
- topic
- year

## Must-Pass Citizen Cases

1. "Cual es el presupuesto total para Villa Maria en 2025?"
Expected: include `74.908.214.520,00` clearly.
2. "Que secretaria ejecuta/seguimiento del presupuesto?"
Expected: `Secretaria de Economia, Transformacion Digital y Desarrollo Productivo` and reference to Art. 9.
3. "Puede cambiar el presupuesto durante el ano?"
Expected: yes, explain it can be modified (Art. 4 y 5).

## Tariff Intent Hints

Treat these intents as amount-sensitive:

- cuanto tengo que pagar
- tasa anual / cuota
- alicuota / porcentaje
- tarifa social
- tributo adicional
- arancel
- costo de servicio municipal

