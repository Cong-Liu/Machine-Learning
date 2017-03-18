import pandas as pd
import numpy as np
import math
#import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_curve
from sklearn import svm

import math
import random, sys
from copy import deepcopy

import Queue

import sys
from random import randint

class Board:
    def __init__(self, player1 = 1, player2 = 2, state = None):
        self.blank = 0
        self.player1 = player1
        self.player2 = player2
        #self.blankNum = 9
        if state is not None:
            #self.state = np.empty_like(state)
            #self.state[:,:] = state
            self.state = state
        else:
            self.state = np.zeros(9).reshape((3, 3))
            self.state = self.state.astype(int)
        pass

    def Put(self, row, column, player = 1):
        #if self.state[row][column] == self.blank:
        if self.IsBlankMove(row, column):
            self.state[row][column] = player
            #self.blankNum -= 1
            return True
        return False

    def IsBlankMove(self, row, col):
        #mrow = move / 3
        #mcol = move % 3
        if self.state[row][col] == self.blank:
            return True
        return False

    def IsOccupiedAll(self):
        #return (self.blankNum == 0)
        
        for i in xrange(0, 3):
            for j in xrange(0, 3):
                if self.state[i][j] == self.blank:
                    return False
        #condlist = [self.state == self.blank]
        #print 'condlist', condlist
        #return True if condlist is None else False
        return True
        pass

    def ResetState(self, state):
        self.state[:,:] = state

        self.blankNum = 0

        for i in xrange(0, 3):
            for j in xrange(0, 3):
                if self.state[i][j] == self.blank:
                    self.blankNum += 1
        pass

    def CheckWinner(self):
        for i in xrange(0, 3):
            if self._CheckCol(i, self.player1):
                return self.player1
            if self._CheckCol(i, self.player2):
                return self.player2
            if self._CheckRow(i, self.player1):
                return self.player1
            if self._CheckRow(i, self.player2):
                return self.player2
        if self._CheckDiagonal(self.player1):
            return self.player1
        if self._CheckDiagonal(self.player2):
            return self.player2
        return 0

    def _CheckRow(self, row, player):
        for i in xrange(0, 3):
            if self.state[row][i] != player:
                return False
        return True

    def _CheckCol(self, col, player):
        for i in xrange(0, 3):
            if self.state[i][col] != player:
                return False
        return True

    def _CheckDiagonal(self, player):
        result = True
        for i in xrange(0, 3):
            result = result and (self.state[i][i] == player)
        if result:
            return True
        result = True
        for i in xrange(0, 3):
            result = result and (self.state[2 - i][i] == player)
        return result

    pass

class Node:
    def __init__(self, board, row = -1, col = -1, parent = None):
        self.board = board
        self.parent = parent
        self.children = []

        self.total = 0.
        self.win = 0.
        self.value = 0.

        self.row = row
        self.col = col

        self.myMove = False

        #self.uctValues = np.zeros(9).astype(int).reshape((3, 3))

        pass

    def GetChild(self, player):
        for i in xrange(0, 3):
            for j in xrange(0, 3):
                if self.board.state[i][j] == 0:
                    b = np.empty_like (self.board.state)
                    b[:,:] = self.board.state
                    b[i][j] = player
                    child = Node(Board(self.board.player1, self.board.player2, b), i, j, self)
                    child.myMove = not self.myMove
                    #print child.row, child.col, child.parent.board.state
                    self.children.append( child )
                    pass
            pass
        return self.children
        pass

    def UpdateNum(self, win, num):
        self.win += win
        self.total += num
        
        pass

    def UpdateValue(self, reward, c = math.sqrt(2)):
        #c = math.sqrt(2)
        if self.total == 0:
            self.total = 1.
        if (self.parent is not None):
            t = self.parent.total
            #print 'self.parent.total', t, np.log(t)
            #self.value = self.win / self.total + c * math.sqrt( 2 * np.log(t) / self.total)
            #self.value = self.win / self.total
            self.value = (reward[self.row][self.col] + self.win) / self.total + c * math.sqrt(np.log(t) / self.total)
            #print self.value
        pass

    def UpdateParentValue(self):
        if self.parent is None:
            return
        ch = self.parent.children
        self.value = 0
        for child in ch:
            self.value += child.value
        pass

    pass

def RandowmOneSet(board, selfplayer):
    #selfplayer = board.state[lastmoveRow][lastmoveCol]
    #print selfplayer, board.state
    opponent = board.player2 if (selfplayer == board.player1) else board.player1

    originalBoard = np.empty_like (board.state)
    originalBoard[:, :] = board.state

    selfMove = False

    while True:
        winner = board.CheckWinner()
        occupied = board.IsOccupiedAll()
        #print 'winner', winner, 'occupied', occupied
        if winner or occupied:
            break;

        while True:
            move = randint(0, 8)
            mrow = move / 3
            mcol = move % 3
            #if board.state[mrow][mcol] == board.blank:
            if board.IsBlankMove(mrow, mcol):
                break
            pass

        board.Put(mrow, mcol, selfplayer if selfMove else opponent)
        #print selfplayer if selfMove else opponent, 'Put ', mrow, mcol
        #print board.state
        selfMove = not selfMove
    
    board.state[:,:] = originalBoard
    #board.ResetState(originalBoard)
    #print originalBoard
    #print board.state
    #print board.blankNum

    if winner == selfplayer:
        #win += 1
        #print selfplayer, 'win '
        return True
    #board.state[:,:] = originalBoard

    #print selfplayer, 'not win '
    return False

def RandomPut(board, selfplayer, loop):
    #selfplayer = board.state[lastmoveRow][lastmoveCol]
    #opponent = board.player2 if (selfplayer == board.player1) else board.player1
    #print 'selfplayer', selfplayer, 'opponent', opponent
    win = 0

    #originalBoard = np.empty_like (board.state)
    #originalBoard[:, :] = board.state
    #print 'originalBoard\n', originalBoard

    for i in xrange(0, loop):

        '''
        selfMove = False

        while True:
            winner = board.CheckWinner()
            if board.IsOccupiedAll() or winner:
                break;

            while True:
                move = randint(0, 8)
                mrow = move / 3
                mcol = move % 3
                #if board.state[mrow][mcol] == board.blank:
                if board.IsBlankMove(mrow, mcol):
                    break
                pass

            board.Put(mrow, mcol, selfplayer if selfMove else opponent)
            print selfplayer if selfMove else opponent, 'Put ', mrow, mcol
            print board.state
            selfMove = not selfMove
        
        if winner == selfplayer:
            win += 1
            print 'win ', win, '/', i
        '''
        IsWin = RandowmOneSet(board, selfplayer)
        if IsWin:
            win += 1
        #print '~~~~~~~~~~win ', win, '/', i + 1, '~~~~~~~~~~~~~~~'

        #board.state[:,:] = originalBoard
        pass

    return win

class SearchTree:
    def __init__(self, player1 = 1, player2 = 2):
        #self.row = row
        #self.column = column
        #self.identity = row * 3 + column
        b = Board(player1, player2)
        #print b.state
        self.root = Node(b)
        #self.root.total = 1.
        self.layer = 3
        
        #self.reward = np.zeros(9).reshape((3, 3))
        #self.reward[self.reward == 0] = 0.9
        self.reward = np.array([[0.5, 0.7, 0.5],[0.7 ,0.9, 0.7],[0.5, 0.7, 0.5]])
        #self.state = state
        #self.parent = parent # "None" for the root node
        #self.children = []

        #self.state[row][column] = 1
        #self.wins = 0
        #self.visits = 0
        #self.avails = 1
        #self.playerJustMoved = playerJustMoved # the only part of the state that the Node needs later
        #print 'state', self.node.state
        pass

    def Search(self, node, player, layer):
        node_iter = node
        selfplayer = player
        opponent = node.board.player2 if (selfplayer == node.board.player1) else node.board.player1
        j = 0
        while True:

            while True:
                # caculate layers
                i = 0
                node_iter2 = node_iter
                if node_iter2 is None:
                    break
                k = 0
                while True:
                    k+= 1
                    if k > 100:
                        print 'k > 100'
                        break
                    node_iter2 = node_iter2.parent
                    i +=1
                    if node_iter2 is None:
                        i -= 1
                        break
                    #if node_iter2.board.blankNum == node.board.blankNum:
                    if node_iter2 is node:
                        break
                if i >= layer:
                    break
                #print '~~~~~~~~~~~~~~~~~~~~~in layer', i

                # if this node has already been expanded
                #print 'len(node_iter.children)', len(node_iter.children)
                if len(node_iter.children) <= 0:
                    isOpponent = (i%2 != 0)
                    self.Expansion(node_iter, opponent if isOpponent else selfplayer, isOpponent)
                
                if node_iter is None:
                    break
                node_iter = self.Selection(node_iter)
                pass

            node_iter = self.Selection(node)
            if node_iter is None:
                break
            if len(node_iter.children) > 0:
                break
            if node_iter.board.IsOccupiedAll():
                break

            j+=1
            if j > 100:
                print 'ERROR: j > 100'
                print 'node\n', node.board.state
                print 'node.BestChild\n', node_iter.board.state
                break

        self.UpdateValues(node)
        uctValues = np.zeros(9).reshape((3, 3))
        for child in node.children:
            uctValues[child.row][child.col] = child.value

        # range of choose: 0-8
        choose = np.argmax(uctValues)
        #print '~!~!~!~~!~!~!~~!~!~!~choose', choose, 'uctValues', uctValues
        return uctValues, choose

    def Selection(self, node):
        if len(node.children) <= 0:
            #print 'ERROR: in Selection node.children', node.children
            #print node.board.state
            selfplayer = node.board.state[node.row][node.col]
            opponent = node.board.player2 if (selfplayer == node.board.player1) else node.board.player1
            children = node.GetChild(opponent)
            if len(node.children) <= 0:
                print 'In Selection, child = 0', node.board.state
                return None
            #for child in children:
            #    child.UpdateNum()
            #return
        self.UpdateValues(node)
        '''
        for child in node.children:
            child.UpdateValue()
        uctValues = np.zeros(9).astype(int).reshape((3, 3))
        for child in node.children:
            uctValues[child.row][child.col] = child.value

        choose = np.argmax(uctValues)
        '''
        choose = node.children[0]
        if choose.myMove:
            for child in node.children:
                if child.value > choose.value:
                    choose = child
        else:
            for child in node.children:
                if child.value < choose.value:
                    choose = child

        #print 'In Selection, choose.row, choose.col, choose.win, choose.total, choose.value', choose.row, choose.col, choose.win, choose.total, choose.value
        #print 'choose.board.state', choose.board.state
        return choose

    def Expansion(self, node, player, isOpponent = False):
        if len(node.children) > 0:
            return
        children = node.GetChild(player)
        #print 'node state', node.board.state
        #print 'children len', len(children)

        for child in children:
            #print 'In Expansion,', child.row, child.col, ' child.board.state', child.board.state
            #win = 1 if RandowmOneSet(child.board, player) else 0
            n = 20
            win = RandomPut(child.board, player, n)
            if isOpponent:
                win = n - win
            child.UpdateNum(win, n)
            #print child.total, child.win, child.value
            '''
            while True:
                child = child.parent
                if child is None:
                    break
                child.UpdateNum(win, n) 
                print child.total, child.win, child.value
                pass
            '''
            self.BackPropogation(child, win, n)
        #print node.total, node.win, node.value
        pass

    def BackPropogation(self, node, win, n):
        while True:
            node = node.parent
            if node is None:
                break
            node.UpdateNum(win, n) 
            #print node.total, node.win, node.value
            pass
        pass

    def UpdateValues(self, node):
        if len(node.children) <= 0:
            return
        for child in node.children:
            child.UpdateValue(self.reward)

        uctValues = np.zeros(9).reshape((3, 3))
        for child in node.children:
            uctValues[child.row][child.col] = child.value

        #print 'in UpdateValues uctValues', uctValues
        #print 'node.board.state', node.board.state
        pass

    def BFS(self):
        queue = Queue.Queue()
        queue.put(self.root)
        i = 0
        while not queue.empty():
            current = queue.get()
            print i, ' queue.get()', current.total, current.win, current.value, '\n', current.board.state
            #children = current.GetChild(1)
            #print 'children:'
            #j = 1
            #for child in children:
            #    print j, '\n', child.board.state
            #    j += 1
            if current.children is not None:
                for child in current.children:
                    queue.put(child)
            i += 1
            #if i > 10:
            #    break;
            pass
        print i
        pass

    def GetNode(self, parent, move):
        # move is 0-8
        #print 'GetNode'
        #print 'parent', parent.board.state
        #print len(parent.children)
        if len(parent.children) < 1:
            print 'ERROR: no children for parent'
            return None
        mrow = move / 3
        mcol = move % 3
        for child in parent.children:
            if (child.row == mrow) and (child.col == mcol) :
                return child
        pass

    '''
    def GetChild1(self):
        for i in xrange(0, self.node.state.shape[0]):
            for j in xrange(0, self.node.state.shape[1]):
                if self.node.state[i][j] == 0:
                    board = np.empty_like (self.node.state)
                    board[:,:] = self.node.state
                    child = Node(i, j, board)
                    child.parent = self
                    self.children.append( child )
                    pass
            pass
        return self.children
        pass

    def PrintChild1(self):
        for x in self.children:
            print 'identity', x.identity
            #print 'parend.id', x.parent.node.identity
            print x.state
        pass
    '''

    pass

def Init(player1 = 1, player2 = 2, layer = 3):
    tree = SearchTree(player1, player2)
    tree.layer = layer
    #uctValues, choose = tree.Search(tree.root, player1, layer)
    #return tree, tree.root, uctValues, choose
    #print tree.root.board.state
    return tree, tree.root

def Run(tree, node, move1, move2):
    if move1 != -1 and move2 != -1:
        player1_move = tree.GetNode(node, move1)
        player2_move = tree.GetNode(player1_move, move2)

        uctValues, choose = tree.Search(player2_move, tree.root.board.player1, tree.layer)
        return uctValues, choose, player2_move
    else:
        uctValues, choose = tree.Search(node, tree.root.board.player1, tree.layer)
        return uctValues, choose, node

def main():
    '''
    Tictactoe = np.array([[0, 1, 2],[3, 4, 5],[6, 7, 8]])
    board = np.zeros(9).reshape((3,3))

    root = Node(1, 1, board)

    mts = Tree(root, board)
    child = mts.GetChild()[0]

    child.Put(1, 0)
    child.GetChild()
    child.PrintChild()

    b = Board(1, 2)

    for i in xrange(0, 3):
        b.Put(2- i, i, 2)
        pass
    win = b.CheckWinner()
    print 'win', win
    tree = SearchTree()
    tree.BFS()
    b = Board()
    root = Node(b)
    children = root.GetChild(1)
    print RandomPut(children[0].board, 0, 0, 10)

    player1 = 1
    player2 = 2
    tree = SearchTree(player1, player2)
    #print 'tree.root.board.state', tree.root.board.state
    #tree.Expansion(tree.root, 1)
    #tree.BFS()
    uctValues, choose = tree.Search(tree.root, player1, 3)
    print 'choose', choose, 'uctValues', uctValues
    
    player1_move = tree.GetNode(tree.root, choose)
    print player1_move.board.state
    choose = GetBlank(player1_move)
    player2_move = tree.GetNode(player1_move, choose)

    uctValues, choose = tree.Search(player2_move, player1, 3)
    
    print 'choose', choose, 'uctValues', uctValues
    player1_move = tree.GetNode(player2_move, choose)
    print player1_move.board.state
    choose = GetBlank(player1_move)
    player2_move = tree.GetNode(player1_move, choose)

    uctValues, choose = tree.Search(player2_move, player1, 3)
    print 'choose', choose, 'uctValues', uctValues
    player1_move = tree.GetNode(player2_move, choose)
    print player1_move.board.state
    choose = GetBlank(player1_move)
    player2_move = tree.GetNode(player1_move, choose)

    uctValues, choose = tree.Search(player2_move, player1, 3)
    print 'choose', choose, 'uctValues', uctValues
    player1_move = tree.GetNode(player2_move, choose)
    print player1_move.board.state
    choose = GetBlank(player1_move)
    player2_move = tree.GetNode(player1_move, choose)
    '''

    tree, node = Init()
    move1 = -1
    move2 = -1
    uctValues, index, node = Run(tree, node, move1, move2)
    print 'move1, move2, index', move1, move2, index
    print 'node\n', node.board.state
    print uctValues
    player1_move = tree.GetNode(node, index)
    move2 = GetBlank(player1_move)
    #print player1_move.board.state

    for x in xrange(0, 4):
        move1 = index
        uctValues, index, node = Run(tree, node, move1, move2)
        print 'move1, move2, index', move1, move2, index
        print 'node\n', node.board.state
        print uctValues
        player1_move = tree.GetNode(node, index)
        move2 = GetBlank(player1_move)


    pass

def GetBlank(node):
    s = node.board.state
    for i in xrange(0, 3):
            for j in xrange(0, 3):
                if s[i][j] == node.board.blank:
                    return i * 3 + j
    pass

if __name__ == "__main__":
    main()