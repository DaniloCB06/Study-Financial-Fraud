!pip install flask-bootstrap
!pip install pyngrok==4.1.1
!pip install flask_bootstrap
!pip install requests-html
!pip install flask-ngrok

!ngrok authtoken NGROK_TOKEN

from google.colab.output import eval_js
print(eval_js("google.colab.kernel.proxyPort(5000)"))

from google.colab import drive
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import numpy as np
import pickle

drive.mount('/content/drive')

template = '/content/drive/MyDrive/templates'

static = '/content/drive/MyDrive/static'

data = '/content/drive/MyDrive/weight_pred_model.pkl'

data2 = '/content/drive/MyDrive/rede_neural.pkl'


app = Flask(__name__, template_folder = template, static_folder = static)
model = pickle.load(open(data2, 'rb'))

run_with_ngrok(app)


@app.route('/')
def teste ():
  return render_template("Meu.html")

@app.route("/teste")
def teste1():
    return render_template("teste.html")


@app.route('/realteste',methods=['POST'])
def teste2 ():
  lista = []
  tipos = ['CASH_IN','CASH_OUT','DEBIT','PAYMENT','TRANSFER']
  real = [str(x) for x in request.form.values()]
  for el in real:
    if el in tipos:
      for i,el1 in enumerate(tipos):
        if el == el1:
          el1 = i
          print(el1)
          lista.append(el1)
    else:
      lista.append(el)
   
  print(real)
  print(lista)
  final_input = [np.array(lista)]
  print(final_input)
  prediction = model.predict(final_input)
  print(prediction)
  if prediction == [1]:
    resultado = 'Fraudou'
  else:
    resultado = 'Não Fraudou'
  
    
  return render_template("Meu.html",output = 'Resultado : {}'.format(resultado))

if __name__ == '__main__':
  app.run()
