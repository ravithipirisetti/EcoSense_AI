"""
EcoSense AI - Species Metadata and Scientific Name Mapping.

Contains mapping dictionary for species codes/common names to scientific names
and helper functions for cleaning species labels.
"""

from typing import Dict, Tuple

# Mapping of common species names to scientific names
SCIENTIFIC_NAME_MAP: Dict[str, str] = {
    "Eurasian Collared Dove": "Streptopelia decaocto",
    "Rose-ringed Parakeet": "Psittacula krameri",
    "Alexandrine Parakeet": "Psittacula eupatria",
    "Plum-headed Parakeet": "Psittacula cyanocephala",
    "Asian Koel": "Eudynamys scolopaceus",
    "Greater Coucal": "Centropus sinensis",
    "Indian Roller": "Coracias benghalensis",
    "White-throated Kingfisher": "Halcyon smyrnensis",
    "Common Kingfisher": "Alcedo atthis",
    "Pied Kingfisher": "Ceryle rudis",
    "House Sparrow": "Passer domesticus",
    "Stork-billed Kingfisher": "Pelargopsis capensis",
    "Asian Green Bee-eater": "Merops orientalis",
    "Blue-tailed Bee-eater": "Merops philippinus",
    "Chestnut-headed Bee-eater": "Merops leschenaulti",
    "Black Drongo": "Dicrurus macrocercus",
    "Greater Racket-tailed Drongo": "Dicrurus paradiseus",
    "White-bellied Drongo": "Dicrurus caerulescens",
    "Red-vented Bulbul": "Pycnonotus cafer",
    "Red-whiskered Bulbul": "Pycnonotus jocosus",
    "White-browed Bulbul": "Pycnonotus luteolus",
    "House Crow": "Corvus splendens",
    "Common Iora": "Aegithina tiphia",
    "Indian Robin": "Copsychus fulicatus",
    "Indian Paradise Flycatcher": "Terpsiphone paradisi",
    "Common Tailorbird": "Orthotomus sutorius",
    "Ashy Prinia": "Prinia socialis",
    "Plain Prinia": "Prinia inornata",
    "Jungle Prinia": "Prinia sylvatica",
    "Jungle Babbler": "Argya striata",
    "Indian Jungle Crow": "Corvus macrorhynchos",
    "Yellow-billed Babbler": "Turdoides affinis",
    "Common Babbler": "Argya caudata",
    "Purple Sunbird": "Cinnyris asiaticus",
    "Purple-rumped Sunbird": "Leptocoma zeylonica",
    "Loten's Sunbird": "Cinnyris lotenius",
    "Baya Weaver": "Ploceus philippinus",
    "Indian Silverbill": "Euodice malabarica",
    "Scaly-breasted Munia": "Lonchura punctulata",
    "White-rumped Munia": "Lonchura striata",
    "Red Avadavat": "Amandava amandava",
    "Black Kite": "Milvus migrans",
    "Brahminy Kite": "Haliastur indus",
    "Shikra": "Accipiter badius",
    "Crested Serpent Eagle": "Spilornis cheela",
    "Black-winged Kite": "Elanus caeruleus",
    "Spotted Owlet": "Athene brama",
    "Jungle Myna": "Acridotheres fuscus",
    "Indian Eagle-Owl": "Bubo bengalensis",
    "Eastern Barn Owl": "Tyto javanica",
    "Indian Grey Hornbill": "Ocyceros birostris",
    "Eurasian Hoopoe": "Upupa epops",
    "Brown-headed Barbet": "Psilopogon zeylanicus",
    "White-cheeked Barbet": "Psilopogon viridis",
    "Coppersmith Barbet": "Psilopogon haemacephalus",
    "Black-rumped Flameback": "Dinopium benghalense",
    "White-naped Woodpecker": "Chrysocolaptes festivus",
    "Streak-throated Woodpecker": "Picus xanthopygaeus",
    "Indian Pitta": "Pitta brachyura",
    "Indian Nightjar": "Caprimulgus asiaticus",
    "Little Swift": "Apus affinis",
    "Red Junglefowl": "Gallus gallus",
    "Indian Peafowl": "Pavo cristatus",
    "Rock Dove": "Columba livia",
    "Spotted Dove": "Spilopelia chinensis",
    "Laughing Dove": "Spilopelia senegalensis",
    "Common Myna": "Acridotheres tristis",
    "Indian Pied Myna": "Gracupica contra",
    "Oriental Magpie-Robin": "Copsychus saularis",
    "White-rumped Shama": "Copsychus malabaricus",
}


def parse_label(raw_label: str) -> Tuple[str, str]:
    """Parse raw label string (e.g., 'S10_Eurasian Collared Dove') into clean species name and scientific name.

    Args:
        raw_label (str): Label string from encoder or dataset.

    Returns:
        Tuple[str, str]: (species_common_name, scientific_name)
    """
    if not raw_label or raw_label.lower() in ("unknown", "unknown bird"):
        return "Unknown Bird", "Unknown"

    parts = raw_label.split("_", 1)
    species_name = parts[1].strip() if len(parts) > 1 else raw_label.strip()

    scientific_name = SCIENTIFIC_NAME_MAP.get(species_name, "Aves sp.")
    return species_name, scientific_name
