**ResNet-50, From-scratch**

- Cặp nhầm nhiều nhất (gộp hai chiều): filet_mignon ↔ steak 16.7%, chocolate_cake ↔ chocolate_mousse 9.5%, apple_pie ↔ bread_pudding 7.4%, prime_rib ↔ steak 7.3%, pork_chop ↔ steak 6.6%
- F1 thấp nhất: steak 0.54, pork_chop 0.61, filet_mignon 0.65, apple_pie 0.65, chocolate_mousse 0.66
- F1 cao nhất: edamame 0.99, oysters 0.96, macarons 0.95

**ResNet-50, Freeze backbone**

- Cặp nhầm nhiều nhất (gộp hai chiều): filet_mignon ↔ steak 14.8%, pork_chop ↔ steak 9.1%, hamburger ↔ pulled_pork_sandwich 8.4%, beef_tartare ↔ tuna_tartare 8.2%, filet_mignon ↔ pork_chop 7.9%
- F1 thấp nhất: steak 0.38, pork_chop 0.41, ravioli 0.41, foie_gras 0.45, scallops 0.46
- F1 cao nhất: edamame 0.98, macarons 0.95, guacamole 0.90

**ResNet-50, Freeze một phần**

- Cặp nhầm nhiều nhất (gộp hai chiều): filet_mignon ↔ steak 13.9%, pork_chop ↔ steak 7.7%, chocolate_cake ↔ chocolate_mousse 7.4%, beef_tartare ↔ tuna_tartare 7.3%, prime_rib ↔ steak 7.3%
- F1 thấp nhất: steak 0.59, pork_chop 0.69, apple_pie 0.71, filet_mignon 0.73, chocolate_mousse 0.74
- F1 cao nhất: edamame 1.00, macarons 0.97, oysters 0.97

**ResNet-50, Full fine-tune**

- Cặp nhầm nhiều nhất (gộp hai chiều): filet_mignon ↔ steak 12.6%, chocolate_cake ↔ chocolate_mousse 9.1%, pork_chop ↔ steak 8.4%, prime_rib ↔ steak 7.9%, beef_tartare ↔ tuna_tartare 7.0%
- F1 thấp nhất: steak 0.61, pork_chop 0.68, chocolate_mousse 0.73, filet_mignon 0.75, apple_pie 0.75
- F1 cao nhất: edamame 1.00, macarons 0.97, spaghetti_carbonara 0.97

**DeiT-S, From-scratch**

- Cặp nhầm nhiều nhất (gộp hai chiều): filet_mignon ↔ steak 14.7%, chocolate_cake ↔ chocolate_mousse 11.8%, samosa ↔ spring_rolls 8.5%, hamburger ↔ pulled_pork_sandwich 8.2%, cheese_plate ↔ panna_cotta 8.1%
- F1 thấp nhất: apple_pie 0.27, bread_pudding 0.32, steak 0.32, breakfast_burrito 0.34, pork_chop 0.35
- F1 cao nhất: edamame 0.97, hot_and_sour_soup 0.88, miso_soup 0.85

**DeiT-S, Freeze backbone**

- Cặp nhầm nhiều nhất (gộp hai chiều): filet_mignon ↔ steak 15.4%, chocolate_cake ↔ chocolate_mousse 11.5%, filet_mignon ↔ pork_chop 10.1%, prime_rib ↔ steak 9.1%, beef_tartare ↔ tuna_tartare 8.6%
- F1 thấp nhất: steak 0.36, pork_chop 0.45, apple_pie 0.46, foie_gras 0.49, filet_mignon 0.51
- F1 cao nhất: edamame 0.98, macarons 0.92, oysters 0.92

**DeiT-S, Freeze một phần**

- Cặp nhầm nhiều nhất (gộp hai chiều): filet_mignon ↔ steak 16.1%, chocolate_cake ↔ chocolate_mousse 8.7%, prime_rib ↔ steak 8.3%, apple_pie ↔ bread_pudding 6.5%, beef_tartare ↔ tuna_tartare 6.5%
- F1 thấp nhất: steak 0.59, pork_chop 0.71, filet_mignon 0.72, apple_pie 0.74, chocolate_mousse 0.75
- F1 cao nhất: edamame 1.00, macarons 0.98, seaweed_salad 0.97

**DeiT-S, Full fine-tune**

- Cặp nhầm nhiều nhất (gộp hai chiều): filet_mignon ↔ steak 14.3%, prime_rib ↔ steak 8.0%, chocolate_cake ↔ chocolate_mousse 7.4%, apple_pie ↔ bread_pudding 6.9%, beef_tartare ↔ tuna_tartare 6.5%
- F1 thấp nhất: steak 0.62, pork_chop 0.70, filet_mignon 0.73, bread_pudding 0.73, apple_pie 0.74
- F1 cao nhất: edamame 0.99, macarons 0.98, spaghetti_carbonara 0.97

