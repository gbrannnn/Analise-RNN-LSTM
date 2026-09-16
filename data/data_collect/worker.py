from datetime import datetime
import json
import os

class Worker:
    def __init__(self, collector):
        self.collector = collector
        self._data = []

    def save_to_file(self, filename):
        path = os.path.join("C:\\Users\\bilol\\Documents\\Projects\\TCC\\data", "raw_data", "pni", filename)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'w', encoding='utf-8') as file:
            json.dump(self._data, file, ensure_ascii=False, indent=2)
    
    def clean_data(self):
        self._data = []

    def run(self):
            print("Starting data collection...")
            offset = 0
            
            date = 2024
            uf_estabelecimento = "SP"
            while True:
                if date == 2027:
                    break

                response = self.collector.collect_data(offset=offset, year=int(date), uf_estabelecimento=uf_estabelecimento)

                if response.status_code != 200:
                    print(f"Erro na requisição: {response.status_code}")
                    date += 1
                    offset = 0
                    print(f"Changing year to {date} and continuing...")
                    continue
    
                content_json = json.loads(response.content)
                parametros = content_json.get("doses_aplicadas_pni")
                if not parametros:
                    date += 1
                    offset = 0
                    print(f"Changing year to {date} and continuing...")
                    continue
    
                self._data.append(parametros)
                # print(f"Offset: {offset}, registros coletados: {len(parametros)}")
                
                if offset % 100000 == 0 and offset != 0:
                    print("saving data to file...")
                    self.save_to_file(f"{date}/data_{offset}.json")  # salva uma única vez no final
                    self.clean_data()  # limpa os dados para a próxima rodada
                
                offset += self.collector.limit
    
            self.save_to_file(f"{date}/data_{offset}.json")  # salva uma única vez no final
            print(f"Total de páginas: {offset // self.collector.limit}, total de registros: {len(self._data)}")
            print("End of data collection.")
            