import json
import hashlib
from pathlib import Path

def assign_split(query: str) -> str:
    h = int(hashlib.sha256(query.strip().encode("utf-8")).hexdigest(), 16)
    return "test" if h % 10 < 3 else "train"

unanswerable_candidates = [
    {
        "id": "unans_001",
        "query": "What are the rules regarding TikTok and Instagram short-form video monetization and revenue sharing?",
        "category": "social_media_monetization",
        "explanation": "None of the indexed 29 agreements contain provisions regarding TikTok, Instagram, or short-form social video monetization."
    },
    {
        "id": "unans_002",
        "query": "Under what circumstances may the licensee use artificial intelligence or generative AI models on licensed footage?",
        "category": "generative_ai_clauses",
        "explanation": "The indexed contracts pre-date generative AI media licensing terms or do not contain AI derivative rights."
    },
    {
        "id": "unans_003",
        "query": "What are the specific penalties for non-compliance with the GDPR Article 33 72-hour breach notification requirement?",
        "category": "data_protection_regulations",
        "explanation": "The contracts do not contain European GDPR Article 33 data incident notification penalty clauses."
    },
    {
        "id": "unans_004",
        "query": "Does this agreement permit cryptocurrency payments such as Bitcoin or Ethereum for quarterly royalty settlement?",
        "category": "cryptocurrency_settlement",
        "explanation": "No contract in the corpus provides for digital asset or cryptocurrency payment settlement mechanisms."
    },
    {
        "id": "unans_005",
        "query": "What are the mandatory carbon offset requirements and Scope 3 emission targets for transportation vehicles?",
        "category": "esg_environmental_targets",
        "explanation": "The transportation and pipeline agreements in the corpus contain no Scope 3 carbon offset or emissions tracking obligations."
    },
    {
        "id": "unans_006",
        "query": "What are the requirements for HIPAA Business Associate Agreements and protected health information encryption standards?",
        "category": "healthcare_compliance",
        "explanation": "Although some pharma/biotech deals exist, none contain HIPAA BAA terms or PHI encryption security schedules."
    },
    {
        "id": "unans_007",
        "query": "What is the agreed procedure for drone aerial photography permits and civil aviation clearance in live broadcasts?",
        "category": "aviation_drone_regulations",
        "explanation": "None of the sponsorship or promotion agreements govern drone flight authorizations or FAA/DGCA clearances."
    },
    {
        "id": "unans_008",
        "query": "How is user data shared with third-party programmatic ad-tracking cookies and pixels regulated under CCPA?",
        "category": "adtech_privacy_regulations",
        "explanation": "The co-branding and web agreements do not contain California Consumer Privacy Act (CCPA) cookie tracking clauses."
    },
    {
        "id": "unans_009",
        "query": "What are the provisions governing strike contingencies and force majeure under SAG-AFTRA or WGA union agreements?",
        "category": "entertainment_union_clauses",
        "explanation": "The indexed contracts do not incorporate SAG-AFTRA or Writers Guild of America theatrical/television collective bargaining terms."
    },
    {
        "id": "unans_010",
        "query": "Does the agreement permit direct-to-home satellite re-transmission in the Indian subcontinent via GSAT transponders?",
        "category": "satellite_broadcast_telecom",
        "explanation": "The contracts do not cover Indian DTH or GSAT satellite frequency transponder allocation."
    },
    {
        "id": "unans_011",
        "query": "What are the terms regarding biometric voice cloning and synthesized vocal likeness for foreign dubbing?",
        "category": "voice_likeness_biometrics",
        "explanation": "The endorsement and licensing agreements have no terms authorizing or restricting voice synthesis or cloning."
    },
    {
        "id": "unans_012",
        "query": "What are the specific minimum guaranteed broadcast hours during prime time on Star Plus or Disney+ Hotstar?",
        "category": "broadcast_linear_quotas",
        "explanation": "None of the contracts mention Indian television linear channels, Star Plus, or Disney+ Hotstar platforms."
    },
    {
        "id": "unans_013",
        "query": "What are the export control requirements under EAR99 and OFAC sanctions compliance for foreign distribution?",
        "category": "sanctions_export_controls",
        "explanation": "The corpus contracts lack specific OFAC economic sanctions screening and EAR99 dual-use re-export controls."
    },
    {
        "id": "unans_014",
        "query": "What are the terms of the key-person life insurance policy required to be maintained on the executive director?",
        "category": "insurance_key_person",
        "explanation": "None of the 29 contracts mandate or define key-person corporate life insurance policies."
    },
    {
        "id": "unans_015",
        "query": "Can the licensor terminate the agreement if the licensee fails to maintain an ESG rating above MSCI BBB?",
        "category": "esg_ratings_covenants",
        "explanation": "There are no ESG rating covenants or termination triggers tied to MSCI sustainability scores in the corpus."
    },
    {
        "id": "unans_016",
        "query": "What are the mandatory COVID-19 safety protocols and remote production vaccination verification procedures?",
        "category": "pandemic_health_protocols",
        "explanation": "The indexed agreements contain no pandemic health testing or COVID-19 vaccination protocols."
    },
    {
        "id": "unans_017",
        "query": "What is the revenue share percentage for non-fungible token (NFT) collectible drops and blockchain assets?",
        "category": "web3_blockchain_assets",
        "explanation": "None of the agreements include NFT, tokenized digital collectibles, or smart contract royalty splitting terms."
    },
    {
        "id": "unans_018",
        "query": "What are the technical mastering standards for Dolby Atmos sound mix delivery and 4K HDR Dolby Vision color grading?",
        "category": "audiovisual_technical_specs",
        "explanation": "The content license and service agreements do not specify Dolby Atmos, HDR10+, or 4K color grade deliverables."
    },
    {
        "id": "unans_019",
        "query": "Does the contract mandate annual third-party penetration testing by a CREST-accredited cybersecurity auditor?",
        "category": "cybersecurity_audits",
        "explanation": "The IT and support agreements do not specify CREST-accredited penetration testing requirements."
    },
    {
        "id": "unans_020",
        "query": "What are the strict compliance procedures for handling children's personal data under COPPA regulations?",
        "category": "child_privacy_regulations",
        "explanation": "None of the co-branding or marketing contracts feature COPPA parental consent verification mandates."
    },
    {
        "id": "unans_021",
        "query": "What is the agreed liquidated damages sum for an unscheduled live OTT streaming interruption exceeding fifteen minutes?",
        "category": "sla_liquidated_damages",
        "explanation": "No contract sets liquidated damages for over-the-top (OTT) digital streaming service interruptions."
    },
    {
        "id": "unans_022",
        "query": "What are the clearance guidelines for synchronization rights in user-generated content (UGC) audio tracks?",
        "category": "music_synchronization_ugc",
        "explanation": "The agreements do not cover sync rights or clearing micro-sync licenses for UGC platforms."
    },
    {
        "id": "unans_023",
        "query": "What are the requirements for local language audio dubbing into Tamil, Telugu, and Malayalam within 48 hours?",
        "category": "regional_language_localization",
        "explanation": "None of the indexed contracts specify Indian regional language dubbing turnarounds (Tamil, Telugu, Malayalam)."
    },
    {
        "id": "unans_024",
        "query": "Under what conditions does the distributor acquire exclusive theatrical distribution rights in IMAX format?",
        "category": "theatrical_imax_distribution",
        "explanation": "The agreements do not grant or specify large-format IMAX theatrical exhibition rights."
    },
    {
        "id": "unans_025",
        "query": "What are the rules governing dynamic CGI virtual product placement inserted into re-runs of licensed programming?",
        "category": "virtual_product_placement",
        "explanation": "The contracts lack clauses governing post-production digital product insertion or dynamic advertising overlay."
    },
    {
        "id": "unans_026",
        "query": "Does the agreement mandate compliance with the EU Digital Services Act illegal content notice-and-action framework?",
        "category": "eu_dsa_regulations",
        "explanation": "None of the contracts reference or adopt the European Union Digital Services Act (DSA) notice mechanisms."
    },
    {
        "id": "unans_027",
        "query": "What penalties apply if the software service vendor fails to achieve SOC 2 Type II compliance certification?",
        "category": "soc2_compliance_penalties",
        "explanation": "The support agreements predate or omit mandatory SOC 2 Type II audit report delivery covenants."
    },
    {
        "id": "unans_028",
        "query": "What are the rights of the parties in the event of an orbital satellite collision or space debris destruction?",
        "category": "space_law_satellite_debris",
        "explanation": "Space debris collision and orbital liability terms are not present in any contract in the corpus."
    },
    {
        "id": "unans_029",
        "query": "What protections govern the digital likeness and post-mortem deepfake recreation of deceased performers?",
        "category": "post_mortem_likeness_rights",
        "explanation": "The endorsement contracts do not contain post-mortem digital recreation, hologram, or deepfake clauses."
    },
    {
        "id": "unans_030",
        "query": "What restrictions apply to utilizing quantum decryption algorithms on encrypted customer databases?",
        "category": "quantum_cryptography_restrictions",
        "explanation": "Quantum decryption restrictions do not exist in any of the indexed contracts."
    },
    {
        "id": "unans_031",
        "query": "What is the compensation protocol for regional cable blackout periods during catastrophic severe weather alerts?",
        "category": "cable_blackout_contingencies",
        "explanation": "None of the contracts address weather emergency regional cable blackout indemnity or fee abatements."
    },
    {
        "id": "unans_032",
        "query": "What are the anti-money laundering (AML) and FinCEN suspicious activity report (SAR) filing thresholds for wire disbursements?",
        "category": "aml_fincen_compliance",
        "explanation": "The contracts do not contain banking-level FinCEN SAR filing thresholds or AML protocols."
    },
    {
        "id": "unans_033",
        "query": "Does this agreement grant exclusive rights to deploy spatial computing interactive apps on the Apple Vision Pro?",
        "category": "spatial_computing_platforms",
        "explanation": "The agreements contain no rights or provisions regarding Apple Vision Pro or spatial headset environments."
    },
    {
        "id": "unans_034",
        "query": "What are the standards for closed-captioning synchronization under FCC CVAA regulations for regional live feeds?",
        "category": "fcc_cvaa_captioning",
        "explanation": "The contracts do not incorporate FCC 21st Century Communications and Video Accessibility Act captioning mandates."
    },
    {
        "id": "unans_035",
        "query": "What is the royalty calculation for music streamed directly into connected autonomous vehicle operating systems?",
        "category": "autonomous_vehicle_infotainment",
        "explanation": "None of the license or promotion contracts address connected automotive infotainment streaming royalties."
    }
]

def main():
    splits = {"train": 0, "test": 0}
    for q in unanswerable_candidates:
        sp = assign_split(q["query"])
        q["split"] = sp
        splits[sp] += 1
    
    print(f"Total unanswerable queries: {len(unanswerable_candidates)}")
    print(f"Split breakdown: train={splits['train']} ({splits['train']/len(unanswerable_candidates)*100:.1f}%), test={splits['test']} ({splits['test']/len(unanswerable_candidates)*100:.1f}%)")

    out_file = Path("data") / "unanswerable_queries.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({
            "description": "Expert-authored unanswerable queries for contract intelligence abstention benchmarking",
            "total_queries": len(unanswerable_candidates),
            "train_queries": splits["train"],
            "test_queries": splits["test"],
            "queries": unanswerable_candidates
        }, f, indent=2, ensure_ascii=False)
    
    print(f"Saved unanswerable queries dataset to {out_file}")

if __name__ == "__main__":
    main()
