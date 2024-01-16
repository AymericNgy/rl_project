To use the checker game code, be sure that :

- you have changed the path for images.png in the file jeu_dames.py in the functions : place_pawns_for_test, place_pawns and test_place_queen

- you can just play by press on a piece and release on another case (respect the rules because it's not so easy sometimes to see allowed moves)


Explanation of the code :

- read the commentary in the code

- I sum up there some points :

My code is working (no bug a priori) for human vs human playing BUT we decided not to use it because : not optimized in term of time and not easily adaptable for Reinforcement Learning. In fact, the algorithm checks if the human move is allowed but the best option would have been to return a list of all the available moves then check is the human move is in it.

Checkers rules need an kind of "IA" to check all the available moves that are not obvious.

This code can be divided in 2 main parts : 

- printing and interacting with the board
- checking if the rules are respected

In the first part, functions :
- create_checkerboard : create a visual render and make this board interactive with canvas
- photo : manage the photos of pawns
- place_pawns_for_test : place pawns/queens like you want, essentially for debug
- place_pawns: place pawns in a the configuration of a real game
- bouger_photo_pion : used to move pieces on board
- clic_appui : get the coordinates when we press the mouse button
- clic_relache : get the coordinates when we release the mouse button

In the second part functions more complex :
- test_place_queen : used to check if a pawn will become a queen
- delete_pawns: at the end of a succession of grasps, remove all the pieces
- pawn_can_eat: check all the possibilities of move on the board with recursivity : redundent code but possible to put the both parts in a common function, return a list of moves of grasping available in this turn
- calcule_max : get the max value for grasping in the list given by pawn_can_eat
- checker : function called when we release the mouse button, it will return True or False (move allowed or not) and tests especially if you can play this color
- checker_with_colors : function called by checker to test more conditions especially if you are in a "grasping mode" that's to say that, during the last turn, your pawn began to take an ennemy piece
- move_de_prise : complex function in which we update the list of available moves when in grasping mode ( removing all the pawns not played in the first turn of grasping mode)
- all_clear : end of grasping mode : clear the lists
- call_simple_depl : simple move if not in grasping mode 