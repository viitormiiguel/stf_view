
import re
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from summarizer import Summarizer

def model_1(article_text):
    
    WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))

    model_name = "csebuetnlp/mT5_multilingual_XLSum"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    input_ids = tokenizer(
        [WHITESPACE_HANDLER(article_text)],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512
    )["input_ids"]

    output_ids = model.generate(
        input_ids=input_ids,
        max_length=84,
        no_repeat_ngram_size=2,
        num_beams=4
    )[0]

    summary = tokenizer.decode(
        output_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False
    )

    print(summary)
    
def model_2(input_text):

    # Create a BERT extractive summarizer
    summarizer = Summarizer()

    # Generate the summary
    summary = summarizer(input_text, min_length=50, max_length=150)  # You can adjust the min_length and max_length parameters

    # Output the summary
    print("Original Text:")
    print(input_text)
    print("\nSummary:")
    print(summary)

if __name__ == '__main__':
    
    
    text = '''Trata-se de embargos de declaração opostos por BANCO DO BRASIL S/A, nos autos do cumprimento de sentença que lhe move JOAQUIM MANOEL
        GRAVATO GERALDES, em face do acórdão que julgou o agravo de instrumento nº  5287315-84.2023.8.21.7000/RS, assim ementado:
        AGRAVO DE INSTRUMENTO. NEGÓCIOS JURÍDICOS BANCÁRIOS. EXPURGOS INFLACIONÁRIOS. CÉDULA RURAL PIGNORATÍCIA. AÇÃO COLETIVA. CUMPRIMENTO DE SENTENÇA.
        VALOR DO LAUDO: R$ 1.240.303,39. FUNDAMENTO: CÉDULAS RURAIS NºS 89/00839-1 E 89/00875-8. ATUALIZAÇÃO MONETÁRIA. O TÍTULO JUDICIAL REFERIU QUE DEVEM SER 
        CORRIGIDOS MONETARIAMENTE OS VALORES A CONTAR DO PAGAMENTO A MAIOR PELOS ÍNDICES APLICÁVEIS AOS DÉBITOS JUDICIAIS (...), 
        QUESTÃO QUE NÃO PODE SER ALTERADA NESTA FASE, POIS ACOBERTADA PELA COISA JULGADA, NO CASO, O ENTENDIMENTO DESTE JULGADOR É DE QUE O ÍNDICE APLICÁVEL 
        AOS DÉBITOS JUDICIAIS É O IGP-M-FORO, POIS INDEXADOR QUE MELHOR REFLETE A CORROSÃO DA MOEDA PELO FENÔMENO INFLACIONÁRIO. A UTILIZAÇÃO DO 
        PROVIMENTO Nº 14/2022-CGJ, PUBLICADO EM 07/04/2022, SOMENTE SERÁ POSSÍVEL EM CARÁTER SUBSIDIÁRIO, OU SEJA, QUANDO INEXISTIR DEFINIÇÃO A 
        RESPEITO NOS AUTOS OU NA LEGISLAÇÃO, O QUE NÃO É O CASO DO PRESENTE FEITO.  NO PONTO, RECURSO DESPROVIDO. JUROS DE MORA – TERMO INICIAL. 
        MESMO EM EXECUÇÕES OU CUMPRIMENTOS DE SENTENÇA INDIVIDUAIS, OS JUROS DE MORA INCIDEM A PARTIR DA CITAÇÃO DO DEVEDOR NO PROCESSO DE CONHECIMENTO DA 
        AÇÃO CIVIL PÚBLICA QUANDO ESTA SE FUNDAR EM RESPONSABILIDADE CONTRATUAL, CUJO INADIMPLEMENTO JÁ PRODUZA A MORA, SALVO A CONFIGURAÇÃO DESTA EM MOMENTO ANTERIOR. 
        ENTENDIMENTO PACIFICADO EM SEDE DE JULGAMENTO REPETITIVO PELO SUPERIOR TRIBUNAL DE JUSTIÇA, NO RESP 1.370.899/SP (TEMA 685 DOS RECURSOS REPETITIVOS), 
        CUJA APLICAÇÃO DEVE SER OBSERVADA EM TODOS OS RECURSOS QUE VENTILEM A MESMA CONTROVÉRSIA. NO PONTO, RECURSO DESPROVIDO. AGRAVO DE INSTRUMENTO DESPROVIDO, POR UNANIMIDADE.
        (TJRS, AGRAVO DE INSTRUMENTO Nº 5287315-84.2023.8.21.7000, 24ª CÂMARA CÍVEL , DESEMBARGADOR JORGE MARASCHIN DOS SANTOS, POR UNANIMIDADE, JULGADO EM 29/11/2023)
        A parte embargante alega que há vícios na decisão recorrida. Sustenta que devem ser aplicados os indíces de correção dos débitos judiciais da Justiça Federal. 
        Argumenta que a aplicação do IGP-M não está prevista na decisão exequenda e acaba por violar a coisa julgada. Afirma que, em se tratando de 
        devedores solidários, não pode haver consequências diferentes sobre a mesma dívida. Pondera ser omisso o acórdão quanto ao fato de que a aplicação 
        do IGP-M implica em onerosidade excessiva ao devedor, bem como sobre a utilização do IPCA em todo o período. Manifesta que os juros de mora devem ser 
        contados desde a citação inicial em cada uma das liquidações e execuções individuais. Prequestiona os dispositivos legais invocados. Pede provimento.'''
    
    # model_1(text)
    
    model_2(text)