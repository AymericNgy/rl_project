# remove bg pour retire le fond des images
import tkinter as tk
from PIL import Image, ImageTk
import numpy as np
import time
class Damier(tk.Tk):
    def __init__(self, taille_case, nb_cases):
        super().__init__()
        self.title("Damier")
        self.taille_case = taille_case
        self.nb_cases = nb_cases
        self.plateau = np.empty((nb_cases, nb_cases), dtype=object)
        self.tour=True # trait aux blancs
        self.liste=[[1,1],[-1,-1],[1,-1],[-1,1]]
        self.length=0
        self.coords_pion_manges=[]
        self.final_pion=[]
        self.final_tour=[]
        self.creer_damier()
        self.coord_appui = None
        self.n=0
        self.mode_prise=False
        self.listemax=[]
        self.listemaxavirer=[]
        # self.placer_pions()
        self.placer_pion_for_test()
        #self.after(1000, self.bouger_pion, 0, 0, 1, 1)
        # self.pack()
        # self.after(1000, self.calcule_max)

    def creer_damier(self):
        for i in range(self.nb_cases):
            for j in range(self.nb_cases):
                couleur = "white" if (i + j) % 2 == 0 else "black"
                canvas = tk.Canvas(self, width=self.taille_case, height=self.taille_case, bd=0, highlightthickness=0, bg=couleur)
                canvas.grid(row=i, column=j)
                canvas.bind("<Button-1>", lambda event, i=i, j=j: self.clic_appui(event, i, j))
                canvas.bind("<ButtonRelease-1>", self.clic_relache)
                self.plateau[i][j]=canvas
                self.plateau[i][j].type="empty"

    def photo(self,image_path,i,j,type):

        image = Image.open(image_path)
        canvas=self.plateau[i][j]
        image = image.resize((self.taille_case, self.taille_case))
        photo = ImageTk.PhotoImage(image)
        canvas.create_image(0, 0, anchor=tk.NW, image=photo)
        canvas.image=photo
        canvas.type=type
        self.plateau[i][j]=canvas

    def placer_pion_for_test(self):

        image_path = r'C:\Users\pujol\OneDrive\Documents\pions_dame\pion_noir.png'
        self.photo(image_path,4,5,"pion_noir")
        self.photo(image_path,6,6,"pion_noir")
        self.photo(image_path,2,5,"pion_noir")
        self.photo(image_path,2,4,"pion_noir")
        self.photo(image_path,2,3,"pion_noir")
        self.photo(image_path,6,3,"pion_noir")
        self.photo(image_path,6,5,"pion_noir")
        image_path = r'C:\Users\pujol\OneDrive\Documents\pions_dame\pion_blanc.png'
        self.photo(image_path,3,2,"pion_blanc")
        self.photo(image_path,5,2,"pion_blanc")

    def placer_pions(self):

        for i in range(self.nb_cases):
            for j in range(self.nb_cases):

                if((i==0 or i==1 or i==2) and(i+j)%2==1):
                    image_path = r'C:\Users\pujol\OneDrive\Documents\pions_dame\pion_noir.png'
                    self.photo(image_path,i,j,"pion_noir")
                elif((i==5 or i==6 or i==7) and(i+j)%2==1):
                    image_path = r'C:\Users\pujol\OneDrive\Documents\pions_dame\pion_blanc.png'
                    self.photo(image_path,i,j,"pion_blanc")

    def delete_pions(self,list):
        for i in list:
            self.plateau[i[0]][i[1]].type="empty"
            self.plateau[i[0]][i[1]].delete("all")


    def bouger_photo_pion(self,x1,y1,x2,y2):
        if(hasattr(self.plateau[x1][y1], 'image')):
            image=self.plateau[x1][y1].image
            self.plateau[x2][y2].create_image(0, 0, anchor=tk.NW, image=image)
            self.plateau[x2][y2].image=image
            self.plateau[x2][y2].type=self.plateau[x1][y1].type
            self.plateau[x1][y1].type="empty"
            self.plateau[x1][y1].delete("all")
        self.update()


    def pion_peut_manger(self, moi,lui,x1,y1): # check toutes les possibilités du tableau, choper le max random parmi les meilleurs
        # si pion noir, on peut manger que les blancs mais dans tous les sens

        self.n+=1

        for l in self.liste:
            # print(l)
            # print(x1+l[0])
            # print(y1+l[1])
            if(x1+l[0]>=1 and x1+l[0]<=6 and y1+l[1]>=1 and y1+l[1]<=6 and self.plateau[x1+l[0]*2][y1+l[1]*2].type=="empty" and self.plateau[x1+l[0]][y1+l[1]].type==lui ):#vérif qu'on est pas au bord : carré interne de 6*6
                # print("x1_int:"+ str(x1))
                # print("y1_int:"+ str(y1))
                # print(self.length)
                self.length+=1 # longeur chaine de prise +=1
                self.coords_pion_manges.append([x1+l[0],y1+l[1]])
                self.plateau[x1+l[0]*2][y1+l[1]*2].type=moi
                self.plateau[x1+l[0]][y1+l[1]].type="empty"
                self.plateau[x1][y1].type="empty"
                self.pion_peut_manger(moi,lui,x1+l[0]*2,y1+l[1]*2) #on rappelle la fonction
                self.coords_pion_manges.remove([x1+l[0],y1+l[1]])
                self.length-=1
                self.plateau[x1+l[0]*2][y1+l[1]*2].type="empty"
                self.plateau[x1+l[0]][y1+l[1]].type=lui
                self.plateau[x1][y1].type=moi
                print("prise")
                # si peut pas manger : on met toutes les positions accumulées dans une liste + la length
        self.final_pion.append([self.coords_pion_manges.copy(),self.length])
    # prendre le pion qui a le plus d'attaque et prendre la meilleure de ses attaques




    def calcule_max(self):
        moi="pion_noir"
        lui="pion_blanc"
        if(self.tour):
            moi="pion_blanc"
            lui="pion_noir"
        for x1 in range (0,8):
            for y1 in range (0,8):
                if(self.plateau[x1][y1].type==moi):
                    self.pion_peut_manger(moi,lui,x1,y1)
                    self.final_tour.append([self.final_pion.copy(),[x1,y1]])
                    self.final_pion=[]
        depl_liste_max=[]
        max=0
        for i in self.final_tour:
            #i[1] sont les coords et i[0] le pion
            for j in i[0]:
                if(j[1]>max):
                    max=j[1]
                    depl_liste_max=[]
                    depl_liste_max.append([j[0],i[1]])
                elif(j[1]==max):
                    depl_liste_max.append([j[0],i[1]])

        return(depl_liste_max)


    # def transfo_dame(self):
        # si dernière rangée MAIS mode prise actif, on n'a pas de dame

    def clic_appui(self, event, ligne, colonne):
        self.coord_appui = (ligne, colonne)

    def clic_relache(self, event):
        x_souris, y_souris = event.x_root-self.winfo_rootx(),event.y_root-self.winfo_rooty()
        # Convertir les coordonnées de la souris en indices de ligne et de colonne
        ligne = y_souris // self.taille_case
        colonne = x_souris // self.taille_case
        if((ligne!=self.coord_appui[0] or self.coord_appui[1]!=colonne) and ligne<=7 and colonne<=7 and ligne>=0 and colonne>=0 and self.checker_pion(self.coord_appui[0],self.coord_appui[1],ligne,colonne)):
            self.bouger_photo_pion(self.coord_appui[0],self.coord_appui[1],ligne,colonne)

    def checker_pion(self,x1,y1,x2,y2):
        print("debut")
        if(self.plateau[x1][y1].type=="pion_blanc"):
            if(not self.mode_prise):
                self.listemax=self.calcule_max()
                print(f"bizarre : ({self.listemax})")

            if(self.listemax[0][0]==[] and not self.mode_prise):

                return False
                 #verif que le deplacement est correct cad la case en diag + la case est vide

            if(not self.mode_prise and self.listemax[0][0]!=[]): # on prend pas encore mais la liste est non vide
                self.mode_prise=True
                #select uniquement le pion concerné par le move
                copie_liste=self.listemax.copy()
                for i in range(0, len(self.listemax)):
                    if(self.listemax[i][1]!=[x1,y1]):# les coords du pion
                        copie_liste.remove(self.listemax[i])
                self.listemax=copie_liste.copy()


            if(self.mode_prise):
                # comparer les coords du pion qu'on veut déplacer pour selctionner les parite de la liste _max
                # compute la destination d'arrivée
                # si mon pion est dans la liste max on peut bouger sinon dégage
                # du coup au début enlever tous les autres pions
                # verif que les coords correspondent au premier élément de la liste puis remove le premier element
                # claucler le point d'arrivée : [x1,y1], i[0][0] : pion noir à sauter, [x2,y2]
                #2*[x1,y1]-i[0][0]
                copie_liste=[]
                autorise=False
                i=0
                while(not autorise and i<len(self.listemax)):
                    coord_predites=np.array([x1,y1])+2*(-np.array([x1,y1])+np.array(self.listemax[i][0][0]))
                    i+=1
                    if(np.all(coord_predites==[x2,y2])):
                        autorise=True
                if(not autorise):
                    return False
                unefois=True
                for i in range(0, len(self.listemax)):
                    print(f"i[0][0] : ({self.listemax[i][0][0]})")
                    coord_predites=np.array([x1,y1])+2*(-np.array([x1,y1])+np.array(self.listemax[i][0][0]))
                    print(f"coord_predites : ({coord_predites})")
                    if(np.all(coord_predites==[x2,y2])):
                        print("lolilollo")
                        print(len(copie_liste))
                        print(i)
                        print(f"avant selflistemax : ({self.listemax})")
                        print(f"avant copie : ({copie_liste})")
                        print(self.listemax[i][0][0])
                        if(unefois):
                            self.listemaxavirer.append(self.listemax[i][0][0])
                            unefois=False
                        copie_liste.append([self.listemax[i][0][1:],self.listemax[i][1]]) # remove toujours la premiere occurence
                        print(f"apres : ({copie_liste})")
                    # else:
                    #     copie_liste.remove(self.listemax[i])
                    #     print(f"i removed : ({self.listemax[i]})")
                    #     print(f"listemax : ({copie_liste})")
                self.listemax=copie_liste.copy()
                if(self.listemax[0][0]==[]):
                    self.mode_prise=False
                    print(f"listeavirer : ({self.listemaxavirer})")
                    self.delete_pions(self.listemaxavirer)
                    self.listemaxavirer=[]
                    self.listemax=[]
                    print("all clear")

                return True

    def get_action(self):



        action=[x1,y1,x2,y2]

        return action

#     def jouer(self,x1,y1,x2,y2):
# # si le réseau donne une mauvaise valeur : relaunch le step jusqu'à temps


if __name__ == "__main__":
    taille_case = 100
    nb_cases = 8
    app = Damier(taille_case, nb_cases)
    app.mainloop()