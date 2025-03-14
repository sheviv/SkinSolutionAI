def get_product_recommendations(condition):
    """Get product recommendations based on skin condition."""
    products_database = {
        "Acne-Prone Skin": [
            {
                "name": "Clear Skin Salicylic Acid Cleanser",
                "description": "Gentle daily cleanser with 2% salicylic acid",
                "price": 24.99,
                "ingredients": [
                    "Salicylic Acid (2%)",
                    "Aloe Vera",
                    "Tea Tree Oil",
                    "Niacinamide",
                    "Glycerin"
                ],
                "key_benefits": [
                    "Unclogs pores",
                    "Reduces inflammation",
                    "Controls excess oil",
                    "Prevents breakouts"
                ],
                "usage_instructions": {
                    "frequency": "Twice daily",
                    "steps": [
                        "Wet face with lukewarm water",
                        "Gently massage cleanser in circular motions",
                        "Leave on for 30 seconds",
                        "Rinse thoroughly"
                    ],
                    "warnings": "May cause initial purging. Start with once daily use."
                },
                "skin_compatibility": ["Oily", "Combination", "Acne-Prone"],
                "ingredient_analysis": {
                    "active_ingredients": {
                        "Salicylic Acid": "A beta hydroxy acid (BHA) that penetrates deep into pores to dissolve excess oil and dead skin cells. Clinically proven to reduce acne and prevent new breakouts. Concentration: 2% - optimal for daily use.",
                        "Tea Tree Oil": "Natural antiseptic with antibacterial properties. Studies show it's effective against P. acnes bacteria. Also helps reduce inflammation and redness.",
                        "Niacinamide": "Form of Vitamin B3 that regulates sebum production, strengthens skin barrier, and reduces inflammation. Works synergistically with other acne-fighting ingredients.",
                        "Aloe Vera": "Contains natural salicylic acid and anti-inflammatory compounds. Soothes irritation and provides lightweight hydration.",
                        "Glycerin": "Humectant that attracts and retains moisture without clogging pores. Helps maintain skin barrier function."
                    },
                    "potential_irritants": {
                        "Salicylic Acid": "May cause dryness or mild irritation initially. Start with once daily use and increase gradually.",
                        "Tea Tree Oil": "Can cause sensitivity in some individuals. Product uses a purified, dermatologist-tested concentration."
                    },
                    "comedogenic_rating": 1,
                    "comedogenic_ingredients": [],
                    "ph_level": "5.5 (Balanced for skin)",
                    "fragrance_free": True,
                    "preservatives": ["Phenoxyethanol", "Ethylhexylglycerin"],
                    "certifications": ["Non-comedogenic", "Dermatologist tested"]
                }
            },
            {
                "name": "Oil-Free Moisturizer",
                "description": "Lightweight, non-comedogenic moisturizer",
                "price": 29.99,
                "ingredients": [
                    "Hyaluronic Acid",
                    "Ceramides",
                    "Zinc PCA",
                    "Green Tea Extract",
                    "Panthenol"
                ],
                "key_benefits": [
                    "Hydrates without clogging pores",
                    "Strengthens skin barrier",
                    "Reduces shine",
                    "Calms irritation"
                ],
                "usage_instructions": {
                    "frequency": "Twice daily",
                    "steps": [
                        "Apply to clean, slightly damp skin",
                        "Gently pat into skin",
                        "Can be layered under sunscreen"
                    ],
                    "warnings": "None"
                },
                "skin_compatibility": ["All Skin Types", "Especially good for Oily and Acne-Prone"],
                "ingredient_analysis": {
                    "active_ingredients": {
                        "Hyaluronic Acid": "A powerful humectant that can hold up to 1000x its weight in water. Different molecular weights penetrate various skin layers for optimal hydration without heaviness.",
                        "Ceramides": "Lipid molecules naturally found in skin barrier. Clinical studies show they improve moisture retention and strengthen skin barrier function.",
                        "Zinc PCA": "A mineral compound that regulates oil production and has antimicrobial properties. Helps reduce shine and prevent breakouts.",
                        "Green Tea Extract": "Rich in polyphenols (EGCG) with proven antioxidant and anti-inflammatory benefits. Helps protect against environmental damage.",
                        "Panthenol": "Pro-vitamin B5 that penetrates skin and converts to pantothenic acid. Proven to improve hydration and reduce inflammation."
                    },
                    "potential_irritants": {},
                    "comedogenic_rating": 0,
                    "comedogenic_ingredients": [],
                    "ph_level": "5.5 (Balanced for skin)",
                    "fragrance_free": True,
                     "preservatives": ["Phenoxyethanol"],
                    "certifications": ["Non-comedogenic", "Allergy tested", "Dermatologist tested"]
                }
            }
        ],
        "Uneven Skin Tone": [
            {
                "name": "Vitamin C Brightening Serum",
                "description": "Advanced formula with 15% vitamin C",
                "price": 45.99,
                "ingredients": [
                    "L-Ascorbic Acid (15%)",
                    "Vitamin E",
                    "Ferulic Acid",
                    "Alpha Arbutin",
                    "Glutathione"
                ],
                "key_benefits": [
                    "Brightens complexion",
                    "Fades dark spots",
                    "Antioxidant protection",
                    "Promotes collagen production"
                ],
                "usage_instructions": {
                    "frequency": "Once daily (morning)",
                    "steps": [
                        "Apply to clean, dry skin",
                        "Wait 30 seconds before next product",
                        "Follow with moisturizer and sunscreen"
                    ],
                    "warnings": "Store in a cool, dark place. May increase sun sensitivity."
                },
                "skin_compatibility": ["All Skin Types"],
                "ingredient_analysis": {
                    "active_ingredients": {
                        "L-Ascorbic Acid": "Pure Vitamin C for brightening and antioxidant protection",
                        "Ferulic Acid": "Stabilizes Vitamin C and provides additional antioxidant benefits",
                        "Alpha Arbutin": "Natural brightening agent"
                    },
                    "potential_irritants": ["L-Ascorbic Acid"],
                    "comedogenic_rating": 0
                }
            },
            {
                "name": "Even Tone Night Cream",
                "description": "Niacinamide and kojic acid complex",
                "price": 39.99,
                "ingredients": [
                    "Niacinamide (5%)",
                    "Kojic Acid",
                    "Licorice Root Extract",
                    "Peptides",
                    "Squalane"
                ],
                "key_benefits": [
                    "Reduces hyperpigmentation",
                    "Evens skin tone",
                    "Improves texture",
                    "Hydrates deeply"
                ],
                "usage_instructions": {
                    "frequency": "Once daily (evening)",
                    "steps": [
                        "Apply to clean skin",
                        "Gently massage until absorbed",
                        "Use as last step in routine"
                    ],
                    "warnings": "Start with alternate days if new to acids."
                },
                "skin_compatibility": ["All Skin Types", "Sensitive"],
                "ingredient_analysis": {
                    "active_ingredients": {
                        "Niacinamide": "Vitamin B3 for brightening and pore reduction",
                        "Kojic Acid": "Natural skin brightener",
                        "Peptides": "Support skin repair and renewal"
                    },
                    "potential_irritants": ["Kojic Acid"],
                    "comedogenic_rating": 1
                }
            }
        ],
        "Dull Skin": [
            {
                "name": "Exfoliating AHA Toner",
                "description": "Gentle exfoliation with glycolic acid",
                "price": 32.99,
                "ingredients": [
                    "Glycolic Acid (7%)",
                    "Lactic Acid (3%)",
                    "Aloe Vera",
                    "Chamomile Extract",
                    "Panthenol"
                ],
                "key_benefits": [
                    "Removes dead skin cells",
                    "Improves radiance",
                    "Smooths texture",
                    "Hydrates"
                ],
                "usage_instructions": {
                    "frequency": "2-3 times weekly",
                    "steps": [
                        "Apply with cotton pad after cleansing",
                        "Avoid eye area",
                        "Follow with moisturizer",
                        "Use sunscreen during day"
                    ],
                    "warnings": "Do not use with other exfoliants. May increase sun sensitivity."
                },
                "skin_compatibility": ["Normal", "Combination", "Oily"],
                "ingredient_analysis": {
                    "active_ingredients": {
                        "Glycolic Acid": "AHA for surface exfoliation",
                        "Lactic Acid": "Gentle AHA for exfoliation and hydration",
                        "Chamomile": "Soothes and calms"
                    },
                    "potential_irritants": ["Glycolic Acid", "Lactic Acid"],
                    "comedogenic_rating": 0
                }
            },
            {
                "name": "Hydrating Essence",
                "description": "Hyaluronic acid and peptide complex",
                "price": 28.99,
                "ingredients": [
                    "Hyaluronic Acid",
                    "Peptides",
                    "Beta-Glucan",
                    "Centella Asiatica",
                    "Panthenol"
                ],
                "key_benefits": [
                    "Deep hydration",
                    "Plumps skin",
                    "Improves absorption of other products",
                    "Soothes and calms"
                ],
                "usage_instructions": {
                    "frequency": "Twice daily",
                    "steps": [
                        "Apply to damp skin after cleansing",
                        "Pat gently until absorbed",
                        "Layer under other products"
                    ],
                    "warnings": "None"
                },
                "skin_compatibility": ["All Skin Types"],
                "ingredient_analysis": {
                    "active_ingredients": {
                        "Hyaluronic Acid": "Hydrates and plumps skin",
                        "Beta-Glucan": "Soothes and supports skin barrier",
                        "Centella Asiatica": "Calming and healing"
                    },
                    "potential_irritants": [],
                    "comedogenic_rating": 0
                }
            }
        ],
        "Healthy Skin": [
            {
                "name": "Maintenance Cleanser",
                "description": "pH-balanced daily cleanser",
                "price": 19.99,
                "ingredients": [
                    "Glycerin",
                    "Panthenol",
                    "Allantoin",
                    "Chamomile Extract",
                    "Green Tea Extract"
                ],
                "key_benefits": [
                    "Gentle cleansing",
                    "Maintains skin barrier",
                    "Soothes and calms",
                    "Non-stripping"
                ],
                "usage_instructions": {
                    "frequency": "Twice daily",
                    "steps": [
                        "Wet face with lukewarm water",
                        "Massage gently for 30-60 seconds",
                        "Rinse thoroughly"
                    ],
                    "warnings": "None"
                },
                "skin_compatibility": ["All Skin Types"],
                "ingredient_analysis": {
                    "active_ingredients": {
                        "Glycerin": "Hydrating humectant",
                        "Panthenol": "Vitamin B5 for soothing and healing",
                        "Allantoin": "Calming and protecting"
                    },
                    "potential_irritants": [],
                    "comedogenic_rating": 0
                }
            },
            {
                "name": "Daily Protection Moisturizer SPF 30",
                "description": "Lightweight sun protection",
                "price": 34.99,
                "ingredients": [
                    "Zinc Oxide",
                    "Titanium Dioxide",
                    "Niacinamide",
                    "Vitamin E",
                    "Green Tea Extract"
                ],
                "key_benefits": [
                    "Broad-spectrum UV protection",
                    "Hydration",
                    "Antioxidant defense",
                    "Non-greasy finish"
                ],
                "usage_instructions": {
                    "frequency": "Daily (morning)",
                    "steps": [
                        "Apply as last skincare step",
                        "Use generous amount",
                        "Reapply every 2 hours if in sun"
                    ],
                    "warnings": "Reapply after swimming or excessive sweating"
                },
                "skin_compatibility": ["All Skin Types"],
                "ingredient_analysis": {
                    "active_ingredients": {
                        "Zinc Oxide": "Physical UV protection",
                        "Niacinamide": "Vitamin B3 for multiple benefits",
                        "Vitamin E": "Antioxidant protection"
                    },
                    "potential_irritants": [],
                    "comedogenic_rating": 1
                }
            }
        ]
    }

    # return products_database.get(condition, products_database["Healthy Skin"])