"""Legal, explicit chess Choice requests from caller-supplied candidate moves."""
import chess


def make_chess_request(fen, playing_side, moves, *, record_id='chess-request'):
    board = chess.Board(fen)
    if not board.is_valid():
        raise ValueError('invalid chess position')
    side = 'white' if board.turn == chess.WHITE else 'black'
    if playing_side.lower() != side:
        raise ValueError('playing side must match the FEN side to move')
    if not 2 <= len(moves) <= 8:
        raise ValueError('provide 2–8 candidate moves')
    options, seen = [], set()
    for text in moves:
        try:
            move = chess.Move.from_uci(text)
        except ValueError:
            move = board.parse_san(text)
        if move not in board.legal_moves:
            raise ValueError(f'illegal candidate move: {text}')
        uci = move.uci()
        if uci in seen:
            raise ValueError('duplicate candidate move')
        seen.add(uci)
        piece = board.piece_at(move.from_square)
        description = f'UCI {uci}; SAN {board.san(move)}; {side} {chess.piece_name(piece.piece_type)} from {chess.square_name(move.from_square)} to {chess.square_name(move.to_square)}'
        if move.promotion:
            description += f', promote to {chess.piece_name(move.promotion)}'
        options.append({'id': uci, 'description': description + '.'})
    ranks = []
    for rank in range(7, -1, -1):
        ranks.append(str(rank + 1) + ': ' + ' '.join(board.piece_at(chess.square(file, rank)).symbol()
                     if board.piece_at(chess.square(file, rank)) else '.' for file in range(8)))
    state = f'Chess position. Side to move: {side}.\nFEN: {board.fen()}\nFiles: a b c d e f g h\n' + '\n'.join(ranks)
    state += '\nUppercase pieces are white; lowercase pieces are black; dot means empty.'
    return {'record_id': record_id, 'state': state, 'questions': [{'id': 'move', 'type': 'choice',
        'instruction': f'Choose the strongest tactical move for {side} from the supplied legal moves.',
        'options': options}]}
