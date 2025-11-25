import os
import csv

def get_spam_stats(csv_path):

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Arquivo não encontrado: {csv_path}")

    spam_count = 0
    ham_count = 0

    with open(csv_path, mode='r', encoding='utf-8-sig') as f:
        leitor = csv.DictReader(f)

        if not leitor.fieldnames or "spam" not in leitor.fieldnames:
            raise ValueError(f"CSV não possui coluna 'spam'. Cabeçalhos encontrados: {leitor.fieldnames}")

        for linha in leitor:
            valor = linha.get("spam")

            if valor is None:
                continue

            try:
                valor_int = int(valor)

                if valor_int == 1:
                    spam_count += 1
                elif valor_int == 0:
                    ham_count += 1

            except ValueError:
                pass

    total = spam_count + ham_count

    spam_pct = (spam_count / total * 100) if total > 0 else 0
    ham_pct  = (ham_count / total * 100) if total > 0 else 0

    return {
        "spam": spam_count,
        "ham": ham_count,
        "total": total,
        "spam_percent": round(spam_pct, 2),
        "ham_percent": round(ham_pct, 2)
    }
