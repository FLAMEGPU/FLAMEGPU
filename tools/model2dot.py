#!/usr/bin/python
# $Id$
#
# Author: Alcione de Paiva Oliveira
# version: 0.9.3
# Date : 21 Oct 2015
#
# Description:
# Create a dot direct graph from the Flame GPU model file. 
# By default the nodes are renamed to avoid direct cycles, so improving readability.
# If the user wants cycles then should use de -c flag
# Optionally, if the user sets the flag -f the script outputs a source code skeleton of the functions used on the Model
#
#---------------------------------------------------------------------
# 
# input arguments
# Example python3 model2dot.py -i XMLModelFile.xml -o out.gdot -f -c


import getopt, sys

from xml.dom.minidom import parse
import xml.dom.minidom

funcOrder =[]
previousState =[]
afterState=[]
conditions=[]
cycle = False
genFunc = False
entrada = ""
saida = ""
mes = []
primeiro = True
clusterNumber = 0

#---------------------------------------------------------------------
#    Get node in text format
#---------------------------------------------------------------------
def getNodeText(node):
    nodelist = node.childNodes
    result = []
    for node in nodelist:
        if node.nodeType == node.TEXT_NODE:
            result.append(node.data)
    return ''.join(result)

#---------------------------------------------------------------------
#    Get condition in text format
#---------------------------------------------------------------------
def getCondition(node):
    globalCond = False
    condT = (node.getElementsByTagName('condition'))
    if (not condT):
       condT = (node.getElementsByTagName("gpu:globalCondition"))
       if (not condT):
           return ""
       else:
           globalCond = True

    lhsNode = node.getElementsByTagName('lhs')[0].getElementsByTagName('agentVariable')
    if (lhsNode):
        lhs = getNodeText(lhsNode[0])
    else:
        lhs = getNodeText(node.getElementsByTagName('lhs')[0].getElementsByTagName('value')[0])

    rhsNode = node.getElementsByTagName('rhs')[0].getElementsByTagName('agentVariable')
    if (rhsNode):
        rhs = getNodeText(rhsNode[0])
    else:
        rhs = getNodeText(node.getElementsByTagName('rhs')[0].getElementsByTagName('value')[0])

    operator = getNodeText(node.getElementsByTagName('operator')[0])
    if (operator == "&lt;"):
        operator = "<"
    elif (operator == "&le;"):
        operator = "<="
    elif (operator == "&gt;"):
        operator = ">"
    elif (operator == "&ge;"):
        operator = ">="

    if (globalCond):
        return "[ label =\"global :"+ lhs+operator+rhs+"\"]"
    return "[ label =\""+ lhs+operator+rhs+"\"]"


#---------------------------------------------------------------------
#    Rename states ocurring more than once
#---------------------------------------------------------------------
def renameStates(agent):
    tam = len(funcOrder)
    for i in range(tam):
       if(previousState[i].find(agent+":")==0 and previousState[i]==afterState[i]):
          afterState[i] = afterState[i]+"\'"
          j = i+ 1
          while (j < tam):
             if (previousState[i] == previousState[j]):
                previousState[j] = previousState[j]+"\'"
             if (previousState[i] == afterState[j]):
                afterState[j] = afterState[j]+"\'"
             j = j+ 1
 

# input arguments
# Example python3 model2dot.py -i XMLModelFile.xml -o out.gdot -f -c

try:
    opts, args = getopt.getopt(sys.argv[1:], "i:o:fc", ["in=", "out="])
except getopt.GetoptError:
    print ('model2dot.py [-f] [-c] -i <model> -o <output>')
    print ('  -c : can have direct cycles')
    print ('  -f : outputs a source code skeleton of the functions used on the Model')
    sys.exit(' ')
for o, a in opts:
    if o in ("-i", "--in"):
        entrada = str(a)
    if o in ("-o", "--out"):
        saida = str(a)
    if o in ("-f"):
        genFunc = True
    if o in ("-c"):
        cycle = True

#-----------------------------------------------------------
#  If is to output source code in C, writes the preamble
#-----------------------------------------------------------
if (len(sys.argv)==4 and sys.argv[3]=="-f" ):
   genFunc = True
   ffunc = open("functions.cc", "w")
   ffunc.write("#ifndef _FUNCTIONS_H_\n#define _FUNCTIONS_H_\n\n#include \"header.h\"\n\n")
   
# Open XML document using minidom parser
DOMTree = xml.dom.minidom.parse(entrada)
collection = DOMTree.documentElement

# Get all the agents in the collection
agents = collection.getElementsByTagName("gpu:xagent")
layers = collection.getElementsByTagName("gpu:layerFunction")

# Get a list of the function execution in the correct order
for layer in layers: 
   nomeFuncLayer = (layer.getElementsByTagName('name')[0]).childNodes[0].data
   funcOrder.append(nomeFuncLayer)
   previousState.append("")
   afterState.append("")
   conditions.append("")
   #print ("Function:%s\n" % (nomeFuncLayer))

file = open(saida, "w")
file.write("digraph model {\n")

# Print detail of each agent.
for agent in agents:
   nome = (agent.getElementsByTagName('name')[0]).childNodes[0].data
   print ("agent: %s" % nome)
   
   if (genFunc):
      ffunc.write("/**************************************\n")
      ffunc.write("     %s functions\n" % (nome))
      ffunc.write("**************************************/\n")

   if not primeiro:
      file.write("}\n\n")

   primeiro = False  
   file.write("subgraph cluster_%d{\n label=\"%s\";color=blue; \n\n" % (clusterNumber,nome))
   clusterNumber = clusterNumber +1
#   memory = agent.getElementsByTagName('memory')[0]
#   variables = memory.getElementsByTagName("gpu:variable")
#   for v in variables:
#      n = v.getElementsByTagName('name')[0]
#      print ("variavel: %s" % n.childNodes[0].data)

   funcs = agent.getElementsByTagName('functions')[0]
   funs = funcs.getElementsByTagName("gpu:function")
   for f in funs:
      noIO =True
      nameFunc = (f.getElementsByTagName('name')[0]).childNodes[0].data
      if (nameFunc not in funcOrder):
         print ('  warning : funtion %s is not in layers list.\n' % (nameFunc))

      print("Function:"+nameFunc)
      condition = getCondition(f)
      rngT = (f.getElementsByTagName('gpu:RNG'))
      if (rngT):
          rng = rngT[0].childNodes[0].data
      else:
          rng = "false"
#      description = f.getElementsByTagName('description')[0]
      ist = (f.getElementsByTagName('currentState')[0]).childNodes[0].data
      fst = (f.getElementsByTagName('nextState')[0]).childNodes[0].data
#      print ("Funcao: %s" % nameFunc)

      # Store the states of each function
      previousState[funcOrder.index(nameFunc)] = nome+":"+ist
      afterState[funcOrder.index(nameFunc)]=nome+":"+fst

      conditions[funcOrder.index(nameFunc)]= condition

      file.write("   %s [shape=box];\n" %  nameFunc )

      if (cycle):
          file.write("   %s -> %s %s;\n" % (ist, nameFunc, condition ))
          file.write("   %s -> %s;\n" %  (nameFunc,  fst))

      outputs = f.getElementsByTagName('outputs')
      if (outputs.length>0):
          gpuOut = outputs[0].getElementsByTagName("gpu:output")[0]
          menName = getNodeText(gpuOut.getElementsByTagName("messageName")[0])
          
          file.write("   %s -> %s [color=green];\n" %  (nameFunc,  menName))

          if (menName not in mes):
              mes.append(menName)

          if (genFunc):
             noIO = False
             if (rng == "false"):
                 ffunc.write("__FLAME_GPU_FUNC__ int %s(xmachine_memory_%s* agent, xmachine_message_%s_list* %s_messages){\n" %(nameFunc,nome,menName,menName))
             else:
                 ffunc.write("__FLAME_GPU_FUNC__ int %s(xmachine_memory_%s* agent, xmachine_message_%s_list* %s_messages, RNG_rand48* rand48){\n" %(nameFunc,nome,menName,menName))
             ffunc.write("//                 add_%s_message(%s_messages, agent->x, agent->y, 0.0,...);\n\n" % (menName, menName))
             ffunc.write("    return 0;\n}\n\n")

      inputs = f.getElementsByTagName('inputs')
      if (inputs.length>0):
          gpuIN = inputs[0].getElementsByTagName("gpu:input")[0]
          menName = getNodeText(gpuIN.getElementsByTagName("messageName")[0])
          
          file.write("   %s -> %s [color=green];\n" %  ( menName, nameFunc))

          if (menName not in mes):
              mes.append(menName)
              
          if (genFunc):
              noIO = False
              if (rng == "false"):
                 ffunc.write("__FLAME_GPU_FUNC__ int %s(xmachine_memory_%s* agent, xmachine_message_%s_list* %s_messages, xmachine_message_%s_PBM* pm){\n" %(nameFunc,nome,menName,menName,menName))
              else:
                 ffunc.write("__FLAME_GPU_FUNC__ int %s(xmachine_memory_%s* agent, xmachine_message_%s_list* %s_messages, xmachine_message_%s_PBM* pm, RNG_rand48* rand48){\n" %(nameFunc,nome,menName,menName,menName))
              ffunc.write("    xmachine_message_%s* current_message = get_first_%s_message(%s_messages, pm, agent->x, agent->y, 0.0);\n" % (menName,menName,menName))
              ffunc.write("    while (current_message)\n")
              ffunc.write("    {\n")
              ffunc.write("       current_message = get_next_%s_message(current_message, %s_messages, pm);\n" % (menName,menName))
              ffunc.write("    }\n")
              ffunc.write("    return 0;\n}\n\n")

      if (genFunc and noIO == True):
          if (rng == "false"):
             ffunc.write("__FLAME_GPU_FUNC__ int %s(xmachine_memory_%s* agent){\n" %(nameFunc,nome))
          else:
             ffunc.write("__FLAME_GPU_FUNC__ int %s(xmachine_memory_%s* agent, RNG_rand48* rand48){\n" %(nameFunc,nome))          
          ffunc.write("    return 0;\n}\n\n")

    #----------------------------------------------------------------------------------
    #  If is not to produce cycles on the same state then the states will be renamed
    #----------------------------------------------------------------------------------
   if (not cycle):
      renameStates(nome)
      for k in range(len(funcOrder)):
          if(previousState[k].find(nome+":")==0 ):
             file.write("   \"%s\" -> %s %s;\n" % (previousState[k][previousState[k].find(":")+1:], funcOrder[k], conditions[k] ))
             file.write("   %s -> \"%s\";\n" %  (funcOrder[k],  afterState[k][afterState[k].find(":")+1:]))

        
#----------------------------------------------------------------------------------
#  Writes the shape of messages
#----------------------------------------------------------------------------------           
file.write("}\n\n")
for m in mes:
    file.write("   %s [shape=box][shape=diamond, label=%s, fontcolor=green, color=green];\n" % (m,m))

file.write("}")

if (genFunc):
   ffunc.write("#endif // #ifndef _FUNCTIONS_H_")
