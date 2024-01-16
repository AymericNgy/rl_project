import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import time
from math import copysign
import random
from copy import deepcopy

class Damier(tk.Tk):

    EMPTY = ' '
    P1 = 'X'
    P2 = 'O'

    def __init__(self, case_size, nb_cases): # init your new checkerboard (size of checkerboard, create a board and place pawns)
        super().__init__()
        self.title("CheckerBoard")
        self.case_size = case_size
        self.nb_cases = nb_cases
        self.checkerboard = np.empty((nb_cases, nb_cases), dtype=object)
        self.turn=True # white turn
        self.length=0
        self.coords_pawns_eaten=[]
        self.final_pawn=[]
        self.final_turn=[]
        self.coord_pressed = None
        self.n=0
        self.grasping_mode=False
        self.listemax=[]
        self.listemax_todelete=[]

        self.create_checkerboard()
        self.place_pawns()


    def create_checkerboard(self): # create a visual render and make this board interactive with canvas
        for i in range(self.nb_cases):
            for j in range(self.nb_cases):
                couleur = "white" if (i + j) % 2 == 0 else "black"
                canvas = tk.Canvas(self, width=self.case_size, height=self.case_size, bd=0, highlightthickness=0, bg=couleur)
                canvas.grid(row=i, column=j)
                canvas.bind("<Button-1>", lambda event, i=i, j=j: self.clic_appui(event, i, j))
                canvas.bind("<ButtonRelease-1>", self.clic_relache)
                self.checkerboard[i][j]=canvas
                self.checkerboard[i][j].type="empty"
                self.checkerboard[i][j].color="empty"

    def photo(self,image_path,i,j,type,color): # manage the photos of pawns

        image = Image.open(image_path)
        canvas=self.checkerboard[i][j]
        image = image.resize((self.case_size, self.case_size))
        photo = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image=photo
        canvas.type=type
        canvas.color=color
        self.checkerboard[i][j]=canvas

    def place_pawns_for_test(self): # place pawns/queens like you want, essentially for debug

        image_path = r'DEFINE IMAGE PATH FOR BLACK PAWNS\pion_noir.png'
        self.photo(image_path,4,5,"pion","noir")
        self.photo(image_path,6,7,"pion","noir")
        self.photo(image_path,2,5,"pion","noir")
        self.photo(image_path,2,4,"pion","noir")
        self.photo(image_path,6,3,"pion","noir")
        self.photo(image_path,6,5,"pion","noir")
        self.photo(image_path,2,3,"pion","noir")
        self.photo(image_path,1,4,"pion","noir")
        image_path = r'DEFINE IMAGE PATH FOR WHITE PAWNS\pion_blanc.png'
        self.photo(image_path,3,2,"pion","blanc")
        self.photo(image_path,5,2,"pion","blanc")

    def place_pawns(self): # place pawns in a the configuration of a real game

        for i in range(self.nb_cases):
            for j in range(self.nb_cases):

                if((i==0 or i==1 or i==2) and(i+j)%2==1):
                    image_path = r'DEFINE IMAGE PATH FOR BLACK PAWNS\pion_noir.png'
                    self.photo(image_path,i,j,"pion","noir")
                elif((i==5 or i==6 or i==7) and(i+j)%2==1):
                    image_path = r'DEFINE IMAGE PATH FOR WHITE PAWNS\pion_blanc.png'
                    self.photo(image_path,i,j,"pion","blanc")

    def test_place_queen(self,x1,y1,x2,y2): # used to check if a pawn will become a queen
        if((self.checkerboard[x1][y1].type=="pion" and x2==0 and self.checkerboard[x1][y1].color=="blanc") or (self.checkerboard[x1][y1].type=="pion" and x2==7 and self.checkerboard[x1][y1].color=="noir")):
            color=self.checkerboard[x1][y1].color
            image_path_dame = r'DEFINE PATH FOR QUEENS IMAGE\dame_'+color+".png"
            self.delete_pawns([[x1,y1]])
            self.photo(image_path_dame,x2,y2,"dame",color)
            self.update()
            return True
        return False



    def delete_pawns(self,list): # at the end of a succession of grasps, remove all the pieces
        for i in list:
            self.checkerboard[i[0]][i[1]].type="empty"
            self.checkerboard[i[0]][i[1]].color="empty"
            self.checkerboard[i[0]][i[1]].delete("all")


    def bouger_photo_pion(self,x1,y1,x2,y2): # used to move pieces on board
        if(hasattr(self.checkerboard[x1][y1], 'image')):
            image=self.checkerboard[x1][y1].image
            self.checkerboard[x2][y2].create_image(0, 0, anchor=tk.NW, image=image)
            self.checkerboard[x2][y2].image=image
            self.checkerboard[x2][y2].type=self.checkerboard[x1][y1].type
            self.checkerboard[x2][y2].color=self.checkerboard[x1][y1].color
            self.checkerboard[x1][y1].type="empty"
            self.checkerboard[x1][y1].color="empty"
            self.checkerboard[x1][y1].delete("all")
        self.update() # work for pawns and queens


    def pawn_can_eat(self,ma_couleur,x1,y1):
    # check all the possibilities of move on the board with recursivity : redundent code but possible to put the both parts in a common function

        if(self.checkerboard[x1][y1].type=="pion"):
            self.n+=1
            liste=[[1,1],[-1,-1],[1,-1],[-1,1]]
            for l in liste:

                if(x1+l[0]>=1 and x1+l[0]<=6 and y1+l[1]>=1 and y1+l[1]<=6 and self.checkerboard[x1+l[0]*2][y1+l[1]*2].type=="empty" and self.checkerboard[x1+l[0]][y1+l[1]].color!=ma_couleur and self.checkerboard[x1+l[0]][y1+l[1]].color!="empty" ):

                    self.length+=1
                    self.coords_pawns_eaten.append([x1+l[0],y1+l[1],x1+2*l[0],y1+2*l[1]])
                    self.checkerboard[x1+l[0]*2][y1+l[1]*2].type=self.checkerboard[x1][y1].type
                    self.checkerboard[x1+l[0]*2][y1+l[1]*2].color=self.checkerboard[x1][y1].color
                    keep_type=self.checkerboard[x1+l[0]][y1+l[1]].type
                    keep_color=self.checkerboard[x1+l[0]][y1+l[1]].color
                    self.checkerboard[x1+l[0]][y1+l[1]].type="empty"
                    self.checkerboard[x1+l[0]][y1+l[1]].color="empty"
                    keep_type2=self.checkerboard[x1][y1].type
                    keep_color2=self.checkerboard[x1][y1].color
                    self.checkerboard[x1][y1].type="empty"
                    self.checkerboard[x1][y1].color="empty"
                    self.pawn_can_eat(ma_couleur,x1+l[0]*2,y1+l[1]*2) #on rappelle la fonction
                    self.coords_pawns_eaten.remove([x1+l[0],y1+l[1],x1+2*l[0],y1+2*l[1]])
                    self.length-=1
                    self.checkerboard[x1][y1].type=self.checkerboard[x1+l[0]*2][y1+l[1]*2].type
                    self.checkerboard[x1][y1].color=self.checkerboard[x1+l[0]*2][y1+l[1]*2].color
                    self.checkerboard[x1+l[0]][y1+l[1]].type=keep_type
                    self.checkerboard[x1+l[0]][y1+l[1]].color=keep_color
                    self.checkerboard[x1+l[0]*2][y1+l[1]*2].type="empty"
                    self.checkerboard[x1+l[0]*2][y1+l[1]*2].color="empty"

                    # if pawn can eat, add the move to a list
            self.final_pawn.append([self.coords_pawns_eaten.copy(),self.length])

        elif(self.checkerboard[x1][y1].type=="dame"): # if a queen, rule to take are different
            liste_des_diags=[[1,1],[-1,-1],[1,-1],[-1,1]]
            liste_des_prises=[]
            liste_des_moves=[]
            for i in range(0, len(liste_des_diags)) :
                l=liste_des_diags[i]
                n=1
                end=False
                while(x1+n*l[0]>=1 and x1+n*l[0]<=6 and y1+n*l[1]>=1 and y1+n*l[1]<=6 and not end):

                    if(self.checkerboard[x1+n*l[0]][y1+n*l[1]].type=="empty"):
                        n+=1

                    else:
                        end=True
                if(end and self.checkerboard[x1+n*l[0]][y1+n*l[1]].color!=ma_couleur):
                    liste_des_prises.append([x1+n*l[0], y1+n*l[1]])
                    n+=1
                    first=True
                    while(x1+n*l[0]>=0 and x1+n*l[0]<=7 and y1+n*l[1]>=0 and y1+n*l[1]<=7 and self.checkerboard[x1+n*l[0]][y1+n*l[1]].type=="empty"):
                        if(first):
                            liste_des_moves.append([])
                            first=False
                        liste_des_moves[-1].append([x1+n*l[0], y1+n*l[1]])
                        n+=1
            self.n+=1
            for m in range(0, len(liste_des_prises)):
                if(liste_des_moves!=[]):
                    for p in range(0,len(liste_des_moves[m])):
                        self.length+=1
                        self.coords_pawns_eaten.append([liste_des_prises[m][0],liste_des_prises[m][1],liste_des_moves[m][p][0],liste_des_moves[m][p][1]])
                        self.checkerboard[liste_des_moves[m][p][0]][liste_des_moves[m][p][1]].type=self.checkerboard[x1][y1].type
                        self.checkerboard[liste_des_moves[m][p][0]][liste_des_moves[m][p][1]].color=self.checkerboard[x1][y1].color
                        keep_type=self.checkerboard[liste_des_prises[m][0]][liste_des_prises[m][1]].type
                        keep_color=self.checkerboard[liste_des_prises[m][0]][liste_des_prises[m][1]].color
                        self.checkerboard[liste_des_prises[m][0]][liste_des_prises[m][1]].type="empty"
                        self.checkerboard[liste_des_prises[m][0]][liste_des_prises[m][1]].color="empty"
                        keep_type2=self.checkerboard[x1][y1].type
                        keep_color2=self.checkerboard[x1][y1].color
                        self.checkerboard[x1][y1].type="empty"
                        self.checkerboard[x1][y1].color="empty"
                        self.pawn_can_eat(ma_couleur,liste_des_moves[m][p][0],liste_des_moves[m][p][1])
                        self.coords_pawns_eaten.remove([liste_des_prises[m][0],liste_des_prises[m][1],liste_des_moves[m][p][0],liste_des_moves[m][p][1]])
                        self.length-=1
                        self.checkerboard[x1][y1].type=self.checkerboard[liste_des_moves[m][p][0]][liste_des_moves[m][p][1]].type
                        self.checkerboard[x1][y1].color=self.checkerboard[liste_des_moves[m][p][0]][liste_des_moves[m][p][1]].color
                        self.checkerboard[liste_des_prises[m][0]][liste_des_prises[m][1]].type=keep_type
                        self.checkerboard[liste_des_prises[m][0]][liste_des_prises[m][1]].color=keep_color
                        self.checkerboard[liste_des_moves[m][p][0]][liste_des_moves[m][p][1]].type="empty"
                        self.checkerboard[liste_des_moves[m][p][0]][liste_des_moves[m][p][1]].color="empty"


            self.final_pawn.append([self.coords_pawns_eaten.copy(),self.length])





    def calcule_max(self,color): # take the max ie the best moves because we must take the option which maximizes the number of taken pieces (the length of a chain of successive pawns)
        for x1 in range (0,8):
            for y1 in range (0,8):
                if(self.checkerboard[x1][y1].color==color):
                    self.pawn_can_eat(color,x1,y1)
                    self.final_turn.append([self.final_pawn.copy(),[x1,y1]])
                    self.final_pawn=[]
        depl_liste_max=[]
        max=0
        for i in self.final_turn:
            for j in i[0]:
                if(j[1]>max):
                    max=j[1]
                    depl_liste_max=[]
                    depl_liste_max.append([j[0],i[1]])
                elif(j[1]==max):
                    depl_liste_max.append([j[0],i[1]])
        self.final_turn=[]
        return(depl_liste_max)




    def clic_appui(self, event, ligne, colonne): # get the coordinates when we press the mouse button
        self.coord_pressed = (ligne, colonne)

    def clic_relache(self, event): # get the coordinates when we release the mouse button
        x_souris, y_souris = event.x_root-self.winfo_rootx(),event.y_root-self.winfo_rooty()
        ligne = y_souris // self.case_size
        colonne = x_souris // self.case_size
        if((ligne!=self.coord_pressed[0] or self.coord_pressed[1]!=colonne) and ligne<=7 and colonne<=7 and ligne>=0 and colonne>=0 and self.checker(self.coord_pressed[0],self.coord_pressed[1],ligne,colonne)):

            if(not self.test_place_queen(self.coord_pressed[0],self.coord_pressed[1],ligne,colonne)):
                self.bouger_photo_pion(self.coord_pressed[0],self.coord_pressed[1],ligne,colonne)

            if(not self.grasping_mode):
                self.turn=not self.turn


    def checker(self,x1,y1,x2,y2): # function called when we release the mouse button, it will return True or False (move allowed or not)
        if(self.turn==True and self.checkerboard[x1][y1].color=="blanc" and self.checkerboard[x2][y2].type=="empty" ):
            return(self.checker_with_colors(x1,y1,x2,y2,"blanc"))
        elif(self.turn==False and self.checkerboard[x1][y1].color=="noir" and self.checkerboard[x2][y2].type=="empty"):
            return(self.checker_with_colors(x1,y1,x2,y2,"noir"))

        return False



    def checker_with_colors(self,x1,y1,x2,y2,couleur): # used in checker to check is the move is allowed
        if(not self.grasping_mode): # if not in a grasping mode yet
            self.listemax=self.calcule_max(couleur) # get the possibilities of grasping
            if(not self.listemax==[] and not self.listemax[0][0]==[]): # grasping list not empty : grasping mode =True because we can take a piece
                self.grasping_mode=True
                return(self.move_de_prise(x1,y1,x2,y2))
            else:
                # grasping list empty : normal move
                return(self.call_simple_depl(x1,y1,x2,y2))
        else: # if already in grasping mode : don't call again the calcule_max function : we must continue with the previous pawn
            return(self.move_de_prise(x1,y1,x2,y2))



    def move_de_prise(self,x1,y1,x2,y2): # used in checker_with_colors to check is the move is allowed

        autorise=False
        i=0
        while(not autorise and i<len(self.listemax)):

            if(np.all(self.listemax[i][0][0][2:]==[x2,y2]) and (np.all(np.array([x1,y1])==np.array(self.listemax[i][1])))): # check the pawn and the move
                autorise=True
            i+=1

        if(not autorise):
            return False
        # select pawn moving
        copie_liste=self.listemax.copy()
        for i in range(0, len(self.listemax)):
            if(self.listemax[i][1]!=[x1,y1]):
                copie_liste.remove(self.listemax[i])
        # change pawn coordinates in the list
        for i in range(0, len(copie_liste)):
            copie_liste[i][1]=[x2,y2]

        self.listemax=copie_liste.copy()
        #listmax has only the moves of one pawn
        copie_liste=[]
        unefois=True

        for i in range(0, len(self.listemax)):
            if(np.all(self.listemax[i][0][0][2:]==[x2,y2])):
                if(unefois):
                    self.listemax_todelete.append(self.listemax[i][0][0])
                    unefois=False
                copie_liste.append([self.listemax[i][0][1:],self.listemax[i][1]])


        self.listemax=copie_liste.copy()
        if(self.listemax==[] or self.listemax[0][0]==[]):
            self.all_clear()
            self.grasping_mode=False
        return True

    def all_clear(self): # a grasping session is terminated so we clear the lists
        self.delete_pawns(self.listemax_todelete)
        self.listemax_todelete=[]
        self.listemax=[]

    def call_simple_depl(self,x1,y1,x2,y2): # if we are not grasping pieces, simply check simple move of pawns/queens

        if(x2-x1==1 and abs(y2-y1)==1 and self.checkerboard[x1][y1].type=="pion" and self.checkerboard[x1][y1].color=="noir"):

            return True
        elif(x2-x1==-1 and abs(y2-y1)==1 and self.checkerboard[x1][y1].type=="pion" and self.checkerboard[x1][y1].color=="blanc"):

            return True

        elif(abs(x2-x1)==abs(y2-y1) and self.checkerboard[x1][y1].type=="dame"): # move diagonally

            diff=[copysign(1,x2-x1), copysign(1,y2-y1)]
            while(x1!=x2):
                x1+=diff[0]
                y1+=diff[1]
                if(self.checkerboard[int(x1)][int(y1)].type!="empty"):
                    return False

            return True
        return False



if __name__ == "__main__":
    case_size = 100
    nb_cases = 8
    app = Damier(case_size, nb_cases)
    app.mainloop()