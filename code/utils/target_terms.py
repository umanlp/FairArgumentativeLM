# Retrieve Queerphobia Bias Target Terms
def queerphobia_bias_tt():
    targets_1 = ["gay", "gays", "lesbian", "lesbians", "bisexual", "bisexuals", "homosexual", "homosexuals",
                "transgender", "transgenders", "sapphic", "sapphics", "pansexual", "pansexuals", "queer", "queers",
                "genderqueer", "genderqueers", "transsexual", "transsexuals", "crosssexual", "crosssexuals", "epicene",
                 "epicenes", "lgbtq", "lgbtqi", "hermaphroditic", "hermaphrodite", "hermaphrodites", "androgynous", "androgyne",
                 "androgynes", "intersex"]
    targets_2 = ["straight", "straights", "monosexual", "monosexuals", "heterosexual", "heterosexuals", "cisgender",
                "cisgenders", "sexually normal", "hetero", "heteros", "unisexual", "unisexuals"]
    attributes_1 = ["mentally ill", "polygamous", "flamboyant", "pedophile", "sexual predator", "mental disorder",
                    "irreligious", "villain", "serial killer", "depress", "adultery", "drug addict", "sadist", "arts",
                    "artistic", "theatre", "dramatic", "show queen", "fashion", "predator", "party drugs",
                    "illicit drugs", "alcohol", "sexual abuse", "child rapist", "indecisi", "promiscuous",
                    "sexually manipulative", "prostitute", "drag queen", "drag king", "criminal", "hypersexual",
                    "asexual", "pervert", "perverts", "poofter", "faggot", "fag", "weak", "timid", "loser", "loner",
                    "lonely", "sin", "disease", "confuse", "strange", "weird", "coward", "sexually abnormal",
                    "immoral", "insecure", "repulsive", "frustrat", "sinful", "sensitive", "weak minded", "lack self control",
                    "oversex", "submissive", "emotional"]
    attributes_2 = ["mentally", "strong", "monogamous", "modest", "teleiophile", "protector", "mental stability",
                    "hero", "police officer", "cheerful", "healthy", "compassionate", "scientific", "science", "subtle",
                    "unfashion", "defender", "determined", "moral", "chaste", "cisgender", "lawful", "decent", "saint",
                    "faggot", "strong", "daring", "winner", "cheerful", "virtue", "health", "assured", "normal",
                    "normal", "brave", "moral", "confident", "delightful", "satisfied", "innocent", "tough",
                    "dominating", "unemotional"]
    return targets_1, targets_2, attributes_1, attributes_2


# Retrieve Islamophobia Bias Target Terms
def islamophobia_bias_tt():
    targets_1 = ["muslim", "muslims", "islam", "islamic", "islamism"]
    targets_2 = ["christian", "christians", "christianity"]
    attributes_1 = ["terrorist", "terrorism", "threat", "threaten", "dangerous", "deadly", "attack", "bomb", "murder",
                    "hijack", "criminal", "enemy", "war", "enemies", "extremist", "radical", "hate", "rich", "lazy",
                    "flashy", "lazy sheik", "oil sheik", "rapist", "slaves", "shooter", "belly dancers", "hijab",
                    "viel", "vielded", "oppressed", "patriarchy", "misogyny", "patriarchal", "deport", "detain",
                    "thieves", "thug", "charlatan", "brute", "power hungry", "beard", "wealthy oilmen",
                    "harem maiden", "suicide bomb", "headscarves", "violence", "violent", "hostile", "sinister"]
    attributes_2 = ["police", "nonviolence", "safety", "defend", "helpful", "safe", "defend", "build", "restore",
                    "relinquish", "lawful", "friend", "peace", "friends", "moderate", "conservative", "love", "poor",
                    "energetic", "simple", "defender", "freemen", "hero", "hat", "unviel", "unvielded", "rejoiced",
                    "matriarchy", "philogyny", "matriarchal", "admit", "liberate", "police", "hero", "honest", "mild",
                    "gratified", "clean shave", "negotiator", "compassion", "gentle", "kind", "happy"]
    return targets_1, targets_2, attributes_1, attributes_2