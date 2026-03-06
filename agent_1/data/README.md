# 1989 Game Boy Longplay Dataset

This dataset contains a collection of Nintendo Game Boy longplay videos across multiple
genres from [World of Longplays](https://longplays.org/).

### RPGs (~48 hrs)
- Pokémon Red - 15:22
- Pokémon Blue - 6:55
- Final Fantasy Legend III - 9:11
- Final Fantasy Legend II - 8:35
- The Final Fantasy Legend - 7:25
- Final Fantasy Adventure - 3:59

### Platformers (~13 hrs)
- Super Mario Land 2: 6 Golden Coins - 2:53
- Super Mario Land 3: Wario Land - 2:46
- Donkey Kong Land III - 2:11
- Donkey Kong Land 2 - 1:58
- Donkey Kong Land - 1:11
- Kirby's Dream Land 2 - 1:47
- Super Mario Land - 0:29

### Action/Adventure (~10 hrs)
- The Legend of Zelda: Link's Awakening - 4:42
- Metroid II: Return of Samus - 1:58
- Donkey Kong (1994) - 1:36
- Gargoyle's Quest - 1:14
- Castlevania II: Belmont's Revenge - 0:53

### Action (~8 hrs)
- Mega Man V - 1:44
- Mega Man IV - 1:27
- Mega Man III - 0:58
- Mega Man II - 0:48
- Mega Man: Dr. Wily's Revenge - 0:40
- Battletoads - 0:20
- Contra: The Alien Wars - 0:24
- R-Type - 0:33
- Ninja Gaiden Shadow - 0:25

### Puzzle/Other (~21 hrs)
- Lemmings - 5:09
- Mario's Picross - 5:45
- Tetris Attack - 2:48
- Tetris Plus - 2:18
- Kirby's Block Ball - 1:41
- Tetris - 0:52
- Kid Icarus: Of Myths and Monsters - 1:28
- Kirby's Dream Land - 0:51

### Estimated Total: ~100 hrs

## Structure

The directory is organized into the following subdirectories:

- `longplays/`: Raw video files downloaded from archive.org.
- `longplays/output/`: Processed video files saved as 160x144 grayscale frames at 30 fps with frame averaging.