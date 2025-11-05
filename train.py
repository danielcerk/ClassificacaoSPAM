import csv
import os
from deep_translator import GoogleTranslator
import time

CSV_ORIGINAL = '/home/kelvin/a3jairo/ClassificacaoSPAM/emails.csv'
CSV_TEMP = '/home/kelvin/a3jairo/ClassificacaoSPAM/emails_temp.csv'

translator = GoogleTranslator(source='en', target='pt')

def traduzir_texto(texto):
    if texto is None or texto.strip() == '' or texto.lower() == 'nan':
        return texto
    try:
        texto_para_traduzir = texto[:4500]
        traducao = translator.translate(texto_para_traduzir)
        if traducao and traducao != texto:
            return traducao
        else:
            return GoogleTranslator(source='auto', target='pt').translate(texto_para_traduzir)
    except Exception as e:
        print(f"Aviso: Erro ao traduzir '{texto[:50]}...': {e}")
        return texto

with open(CSV_ORIGINAL, mode='r', encoding='utf-8', newline='') as arquivo_entrada, \
     open(CSV_TEMP, mode='w', encoding='utf-8-sig', newline='') as arquivo_saida:

    leitor = csv.DictReader(arquivo_entrada)
    campos = leitor.fieldnames
    if campos is None:
        raise ValueError("Arquivo CSV está vazio ou sem cabeçalho válido.")

    escritor = csv.DictWriter(arquivo_saida, fieldnames=campos)
    escritor.writeheader()

    linha_num = 0
    for linha in leitor:
        linha_num += 1
        for coluna, valor in linha.items():
            if valor is not None and isinstance(valor, str):
                linha[coluna] = traduzir_texto(valor)
                time.sleep(0.3)  
        escritor.writerow(linha)
        if linha_num % 10 == 0:
            print(f"Traduzidas {linha_num} linhas...")

os.replace(CSV_TEMP, CSV_ORIGINAL)

print(f"\nTradução completa Arquivo atualizado:\n{CSV_ORIGINAL}")
