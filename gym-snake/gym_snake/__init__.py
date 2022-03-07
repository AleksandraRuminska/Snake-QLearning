from gym_snake.register import register

for num_players in ['', '2s', '3s', '4s']:
    for style in ['DeadApple', '', 'Expand', '4a']:
        for grid_size in ['4x4', '8x8', '16x16']:
            for grid_type in ['', 'Hex']:
                env_id = '-'.join(['Snake', grid_type, grid_size, style, num_players]) + '-v0'.replace('--', '-')
                entry_point = 'gym_snake.envs:' + '_'.join(['Snake', grid_type, grid_size, style, num_players]).replace('--', '-')
                print("register(")
                print("    id='" + env_id + "',")
                print("    entry_point='" + entry_point + "'")
                print(")")

                pass

register(
    id='Snake-16x16-v0',
    entry_point='gym_snake.envs:Snake_16x16'
)
