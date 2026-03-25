# position_commentary.py

import random


def describe_position(features, eval_bars):
    """
    Generate detailed, human-like commentary on a chess position, including move difficulty
    and overall position assessment. Non-repetitive and cause-effect sentences.

    Args:
        features (dict): Output of compute_features(board, engine)
        eval_bars (dict): {"position_quality": float, "move_ease": float}
    Returns:
        str: Detailed commentary
    """

    commentary = []

    # --- Overall position quality ---
    pq = eval_bars.get("position_quality", 0)
    if pq > 5:
        position_quality_desc = random.choice([
            "The overall position is extremely favorable, giving you a strong advantage.",
            "You are clearly dominating the board, and your pieces are in excellent positions.",
            "The position is very advantageous, allowing you to dictate play."
        ])
    elif pq > 1:
        position_quality_desc = random.choice([
            "The position is slightly better for you, with some advantages you can exploit.",
            "You have a comfortable edge in the position, with opportunities to improve further.",
            "Things are going well for your side, though careful play is still required."
        ])
    elif pq > -1:
        position_quality_desc = random.choice([
            "The position is roughly balanced; neither side has a clear advantage.",
            "Both sides have chances, and small errors could tilt the game.",
            "The position is neutral, requiring accurate moves to maintain equality."
        ])
    elif pq > -5:
        position_quality_desc = random.choice([
            "The position is slightly worse for you, and some pressure needs careful handling.",
            "You are under moderate pressure, requiring precise moves to avoid falling behind.",
            "Things are not ideal, but the game is still playable with caution."
        ])
    else:
        position_quality_desc = random.choice([
            "The position is very unfavorable, and your opponent has a clear advantage.",
            "You are in serious trouble, with most of the position controlled by the opponent.",
            "The board situation is extremely difficult, and mistakes could be costly."
        ])
    commentary.append(position_quality_desc)

    # --- Move difficulty ---
    me = eval_bars.get("move_ease", 0)
    if me > 3:
        move_desc = random.choice([
            "The next move is easy to find, and you have several good options available.",
            "Finding the correct move is straightforward, allowing you to play confidently.",
            "The position allows simple and effective moves without complex calculations."
        ])
    elif me > 0:
        move_desc = random.choice([
            "The next move requires some thought, with a few options that need careful evaluation.",
            "You will need to calculate accurately to find a satisfactory move.",
            "There are tricky choices ahead; one should proceed cautiously."
        ])
    else:
        move_desc = random.choice([
            "The next move is very hard to find, and mistakes are likely if not calculated carefully.",
            "This is a complex position where finding the right move is challenging.",
            "The position is tricky, and selecting a correct move requires deep thought."
        ])
    commentary.append(move_desc)

    # --- Mobility & Space Control ---
    mobility = features.get("mobility", 0)
    space = features.get("space_control", 0)
    if mobility > 12 and space > 5:
        mobility_space_desc = random.choice([
            "Your pieces are extremely active, and you control a lot of space, which gives you freedom to maneuver and create threats.",
            "High mobility combined with space dominance allows you to coordinate attacks effectively and dictate play.",
            "With such piece activity and control over territory, you can execute your plans comfortably."
        ])
    elif mobility > 5 and space > 0:
        mobility_space_desc = random.choice([
            "Your pieces have decent mobility, and there is moderate control over space, giving you flexible options.",
            "You can move your forces without much restriction, though some areas remain contested.",
            "There is reasonable freedom for your pieces to operate, allowing strategic maneuvering."
        ])
    else:
        mobility_space_desc = random.choice([
            "Your pieces are somewhat restricted, and space is limited, which constrains your options and makes planning harder.",
            "Cramped mobility and tight space reduce your ability to coordinate attacks and defend effectively.",
            "Limited movement and lack of territory control create challenges in finding effective plans."
        ])
    commentary.append(mobility_space_desc)

    # --- King Safety & Traps ---
    king_exp = features.get("king_exposure", 0)
    trap = features.get("trap_susceptibility", 0)
    volatility = features.get("volatility", 0)
    king_desc = ""
    if king_exp > 2:
        king_desc = random.choice([
            "Your king is somewhat exposed, making tactical vigilance necessary.",
            "King safety is a concern, and threats may appear quickly.",
            "Watch out for potential attacks on your king, as defenses are limited."
        ])
    else:
        king_desc = random.choice([
            "Your king is well-protected and safe from immediate threats.",
            "King safety is solid, allowing you to focus on other strategic objectives.",
            "The king is not under direct danger, providing stability in your position."
        ])
    if trap > 0.3 or volatility > 0.5:
        trap_desc = random.choice([
            "Additionally, the position contains tactical traps and potential pitfalls, so careful calculation is required.",
            "There are hidden dangers in your moves, and inaccurate play may be punished.",
            "Tactical complexity is high; precise moves are crucial to avoid blunders."
        ])
        commentary.append(f"{king_desc} {trap_desc}")
    else:
        commentary.append(king_desc)

    # --- Material & Pawn Structure with cause-effect ---
    mat = features.get("material_imbalance", 0)
    pawn_struct = (features.get("pawn_majority", 0)
                   - features.get("doubled_pawns", 0)
                   - features.get("backward_pawns", 0))

    if mat > 2:
        mat_desc = "You are ahead in material, giving you a tangible advantage."
        if pawn_struct < -1:
            pawn_desc = "However, your pawn structure has weaknesses that may limit the effectiveness of this material edge."
        else:
            pawn_desc = "Moreover, your pawns are well-placed, complementing your material advantage."
    elif mat < -2:
        mat_desc = "You are behind in material, which increases pressure on your play."
        if pawn_struct > 1:
            pawn_desc = "However, your pawn structure is strong and may help compensate for the material deficit."
        else:
            pawn_desc = "Additionally, weak pawn structure exacerbates your material disadvantage."
    else:
        mat_desc = "Material is roughly balanced between the sides."
        if pawn_struct > 1:
            pawn_desc = "Your pawn structure gives a slight positional edge."
        elif pawn_struct < -1:
            pawn_desc = "Pawn weaknesses slightly hinder your options."
        else:
            pawn_desc = "Pawn structure is standard, with no significant weaknesses."
    commentary.append(f"{mat_desc} {pawn_desc}")

    # --- Advice based on difficulty and evaluation ---
    advice = []
    if me < 0:
        advice.append(random.choice([
            "Careful calculation is necessary; the next move is difficult and errors are costly.",
            "Think ahead carefully to find the correct move and avoid tactical mistakes.",
            "Pay attention to threats and plan multiple candidate moves before acting."
        ]))
    if pq < -2:
        advice.append(random.choice([
            "Consider defensive moves and improving coordination to reduce pressure.",
            "Simplifying the position may help mitigate the disadvantage.",
            "Try to consolidate your pieces to stabilize the position."
        ]))
    if pq > 3 and me > 2:
        advice.append(random.choice([
            "You can play actively to increase your advantage.",
            "Look for tactical opportunities to press your lead.",
            "Capitalize on your strong position while moves are easy to find."
        ]))
    if advice:
        commentary.append(" ".join(advice))

    return " ".join(commentary)
