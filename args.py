def get_args(parser):
    # Generic hyperparamters
    parser.add_argument("--num_epochs", type=int, default=10)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--num_layers", type=int, default=3)
    parser.add_argument("--sanity_check", action="store_true")  # acts like a flag
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--embed_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--device", type=str, default='mps')
    parser.add_argument("--from_pretrained", action="store_true")
    parser.add_argument("--visualize", action="store_true")

    args = parser.parse_args()