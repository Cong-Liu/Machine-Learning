from Tkinter import *
import random
from random import randint, shuffle
import tkFont
import numpy as np

import mcts

tree, node = mcts.Init()


class BoardGame:
    """This class represent a game board window"""

    def __init__(self, size = 3):
        root = Tk()
        self.window = root
        self.size = size
        self.values = []
        self.players = []
        self.moves = []
        self.status = -1 #game status, -1 means tie game, 0 means player 0 win, 1 means player 1 win
        cellSize = 100
        #set window title and size
        root.wm_title("Board Game")
        root.geometry('{}x{}'.format(size * cellSize, size * cellSize))

        #create lable to display messages
        self.label=Label(root, text="Welcome! Click New to start", font="Times 16 bold")
        self.label.grid(row = 0, column = 0, columnspan = size, sticky = 'EWN')

        #create buttons
        self.buttons = self.initButtons(1)

        #create menu bar
        self.initMenu()

        self.clickedButton = IntVar()

    def say_hi(self):
        print "This function is not implemented."

    def run(self):
        self.window.mainloop()

    def initMenu(self):
        menubar = Menu(self.window)
        filemenu = Menu(menubar,tearoff=0)
        filemenu.add_command(label="New", command=self.reset)
        filemenu.add_command(label="Save", command=self.say_hi)
        filemenu.add_separator()
        filemenu.add_command(label="Exit", command=self.quit)
        menubar.add_cascade(label="File", menu=filemenu)
        helpmenu = Menu(menubar, tearoff=0)
        helpmenu.add_command(label="About", command=self.about)
        menubar.add_cascade(label="Help", menu=helpmenu)
        self.window.config(menu=menubar)
    
    def initButtons(self, startRow):
        font = tkFont.Font(family = 'Helvetica', size = 36)
        buttons = []
        index = 0
        for row in range(0 + startRow, self.size + startRow):
            for col in range(0, self.size):
                button = Button(self.window,
                               text = str(index),
                               font = font)
                button.grid(row = row, column = col, sticky='EWNS')
                button.row = row - startRow
                button.col = col
                button.index = index
                index = index + 1
                buttons.append(button)
                button.bind("<Button-1>", self.onClick)
                self.window.columnconfigure(col, weight=1)
                self.values.append(-1)
            self.window.rowconfigure(row, weight=1)
        return buttons

    def onClick(self, event):
        if event.widget["state"] == DISABLED: return
        print event.widget.row, event.widget.col, event.widget.index
        self.clickedButton.set(event.widget.index)

    def about(self):
        print "This program is made by Cong Liu"

    def reset(self):
        """Reset the board and start a new game"""
        self.clickedButton.set(-1)

        for button in self.buttons:
            button.config(text = "  ", state = NORMAL)
            self.values[button.index] = -1

        self.showMessage("New game started.")
        if len(self.players) == 0:
            raise RuntimeError("No player found")
        else:
            self.play()

        self.moves = []
        global tree
        global node

        node = tree.root

    def showMessage(self, msg):
        self.label.config(text = msg)

    def setCell(self, row, col, text, disable = False):
        if row < 0 or row >= self.size:
            raise Exception("Row error")
        elif col < 0 or col >= self.size:
            raise Exception("Column error")
        
        state = NORMAL
        if disable: state = DISABLED
        self.buttons[self.size * row + col].config(text = text, state = state)

    def addPlayer(self, player):
        player.id = len(self.players)
        self.players.append(player)

    def play(self):
        index = 0
        max = self.size * self.size
        self.status = 0
        while index < max:
            current = index % len(self.players)
            player = self.players[current]
            self.showMessage("Player " + str(current + 1) + " move.")
            move = player.move()
            if move < 0: return    #game stopped
            self.setCell(move[0], move[1], player.chess, True)
            self.values[move[0] * self.size + move[1]] = current
            index = index + 1
            self.moves.append(move)
            win = self.checkGameOver(move)
            if win != 0: break
        if win <= 0:
            self.showMessage("Game over. Tie.")
            self.status = -1
        else:
            self.showMessage("Game over. Player " + str(win) + " win!")
            self.status = win-1

    def checkGameOver(self, move):
        """Check if the game is over. Return 0 means game is not over, return -1 means tie, otherwise return which player is win"""
        player = self.values[move[0] * self.size + move[1]]
        vl = np.reshape(self.values,(self.size, self.size))
        pl = vl == player
        #check row
        sumrow = np.sum(pl,0)
        #check col
        sumcol = np.sum(pl,1)
        #check angnal
        sum1, sum2 = 0, 0
        for i in range(0, self.size):
            sum1 += pl[i][i]
            sum2 += pl[i][self.size - 1 - i]

        if self.size == sum1 or self.size == sum2:
            return player + 1

        templist = sumcol.tolist() + sumrow.tolist()
        if self.size in templist:
            return player + 1

        return 0

    def quit(self):
        self.clickedButton.set(-1)
        self.window.quit()
        exit()



class Player(object):
    def __init__(self, board, ch):
        self.board = board
        self.chess = ch

    def move(self):
        print "Player moved"

class Human(Player):
    def move(self):
        self.board.window.wait_variable(self.board.clickedButton)
        index = self.board.clickedButton.get()
        if index < 0: return index
        row = index / self.board.size
        col = index % self.board.size
        return row, col

class EasyAI(Player):
    def move(self):
        max = len(self.board.values) - 1
        while True:
            index = randint(0, max)
            if self.board.values[index] == -1: break

        row = index / self.board.size
        col = index % self.board.size
        return row, col

class MiddleAI(Player):
    def move(self):
        if self.board.size !=3: raise NotImplementedError("Middle AI only suitable for 3x3 board.")
        center = [4]
        boarder = [1, 3, 5, 7]
        shuffle(boarder)
        corner = [0, 2, 6, 8]
        shuffle(corner)
        choices = center + boarder + corner
        for index in choices:
            if self.board.values[index] == -1: break

        row = index / self.board.size
        col = index % self.board.size

        # set fake reword r on each move.
        val = self.board.values
        for idx in range(len(val)):
            if val[idx] is -1:
                r = "{0:.2f}%".format(random.random())
                self.board.setCell(idx/self.board.size, idx%self.board.size, r )
        return row, col

class HighAI(Player):
    #tree, node = mcts.Init()
    #mcts

    #self.board: board game class
    #self.board.values: list of length 9, possible values: -1 means this cell is empty, 0 means player 0, 1 means player 1
    #self.board.setCell(row, col, text): show text on a cell
    #self.board.moves: list of all moves. each move is a triple [row, col]
    #self.board.status: the game status, -1 means tie game, 0 means player 0 win, 1 means player 1 win

    def move(self):
        global tree
        global node

        #print tree.layer
        movelist = self.board.moves
        listlen = len(movelist)
        if listlen <2:
            move1 = -1
            move2 = -1
        else:
            move = movelist[listlen-2]
            move1 = move[0] * 3 + move[1]
            move = movelist[listlen-1]
            move2 = move[0] * 3 + move[1]
        uctValues, index, node = mcts.Run(tree, node, move1, move2)
        print 'move1, move2, index', move1, move2, index
        #print 'node', node.board.state
        #print uctValues
        row = index / self.board.size
        col = index % self.board.size

        uctValues = uctValues.reshape(9)
        # set fake reword r on each move.
        val = self.board.values
        for idx in range(len(val)):
            if val[idx] is -1:
                r = format(uctValues[idx],'.1%')
                print uctValues[idx]
                self.board.setCell(idx/self.board.size, idx%self.board.size, r )

        return row, col


#test main function
game = BoardGame(3)
game.addPlayer(HighAI(game, "O"))
game.addPlayer(Human(game, "X"))
#tree, node = mcts.Init()
game.reset()
#game.values
game.run()


