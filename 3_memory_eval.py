"""记忆能力评测：本轮固定7题的LoCoMo词元F1，单文件离线评分。

依赖：python -m pip install regex==2026.9.29 nltk==3.9.2
用法：python memory_f1_eval.py --predictions predictions.jsonl --gold gold.json --output f1_results.json
predictions.jsonl每行：{"cell":"codex_qwen35_9b","task_id":"conv-26-q050","run_id":"唯一运行ID","status":"response_received_ungraded","prediction":"模型完整回答"}
gold.json为对象列表，每项包含task_id、answer、tag；使用已有评分端参考答案。
四个组合各7题；缺题或重复题会报错；超时/失败/空回答保留为0分。
不运行模型，不做语义打分，不截短模型回答。仅覆盖本轮非对抗事实/时间题。
参考答案和此评分程序仅留在操作者评分端，不作为被测Agent输入。

上游：Maharana等，Evaluating Very Long-Term Conversational Memory of LLM Agents，ACL2024。
https://github.com/snap-research/locomo/blob/3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376/task_eval/evaluation.py
normalize_answer与f1_score原样摘录自该固定提交；本文件新增输入校验和分组均值。
上游许可证：CC BY-NC 4.0，https://creativecommons.org/licenses/by-nc/4.0/
完整上游许可文本附于文件末尾。当前7题整理标签1/2对应官方类别4/2，均使用f1_score。
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import string
import sys
import regex
from nltk.stem import PorterStemmer

ps = PorterStemmer()
TASK_IDS = ('conv-26-q050','conv-26-q051','conv-26-q052','conv-26-q053','conv-26-q000','conv-26-q004','conv-26-q005')
CELLS = ('codex_qwen35_9b','codex_gemma4_e4b','dsh_qwen35_9b','dsh_gemma4_e4b')
FAILED = {'timeout','execution_error','execution_failure','no_final_response','empty_response','parse_error'}

def normalize_answer(s):

    s = s.replace(',', "")
    def remove_articles(text):
        # return regex.sub(r'\b(a|an|the)\b', ' ', text)
        return regex.sub(r'\b(a|an|the|and)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def f1_score(prediction, ground_truth):
    prediction_tokens = [ps.stem(w) for w in normalize_answer(prediction).split()]
    ground_truth_tokens = [ps.stem(w) for w in normalize_answer(ground_truth).split()]
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    # print('# F1 #', prediction, ' | ', ground_truth, ' #', precision, recall, f1)
    # return recall
    return f1

def evaluate(predictions, gold):
    refs = {}
    for row in gold:
        task = row['task_id']
        if task in refs: raise ValueError('重复参考答案：' + task)
        refs[task] = row
    if not set(TASK_IDS).issubset(refs): raise ValueError('缺少固定7题的参考答案')
    indexed = {}
    for row in predictions:
        key = (row['cell'], row['task_id'])
        if key[0] not in CELLS or key[1] not in TASK_IDS: raise ValueError('超出固定矩阵：' + str(key))
        if key in indexed: raise ValueError('重复回答：' + str(key))
        if not isinstance(row.get('run_id'), str) or not row['run_id'].strip(): raise ValueError('缺少run_id')
        if row.get('status') not in FAILED | {'response_received_ungraded'}: raise ValueError('未完成或未知状态：' + str(key))
        if not isinstance(row.get('prediction'), (str, type(None))): raise ValueError('回答必须为文本或null')
        indexed[key] = row
    expected = {(c,t) for c in CELLS for t in TASK_IDS}
    if set(indexed) != expected: raise ValueError('矩阵未完整：四组各7题，不允许静默剔除缺题')
    cases, summary = [], []
    for cell in CELLS:
        scores = []
        for task in TASK_IDS:
            row = indexed[(cell,task)]; ref = refs[task]
            if ref.get('tag') not in (1,2): raise ValueError('本文件只覆盖本轮事实/时间题')
            if not isinstance(ref.get('answer'), str): raise ValueError('参考答案必须为文本')
            answer = row.get('prediction') or ''
            failed = row['status'] in FAILED or not answer.strip()
            value = 0.0 if failed else float(f1_score(answer, ref['answer']))
            scores.append(value)
            cases.append({'cell':cell,'task_id':task,'run_id':row['run_id'],'status':row['status'],'prediction':answer,'f1':value})
        summary.append({'cell':cell,'n_expected':len(TASK_IDS),'official_f1_mean':sum(scores)/len(TASK_IDS)})
    return {'metric':'LoCoMo official token F1; macro mean over fixed7 tasks','scope':'conv-26 full-history pilot; not full benchmark or persistent memory','upstream_commit':'3eb6f2c585f5e1699204e3c3bdf7adc5c28cb376','cases':cases,'summary':summary}


def main():
    if hasattr(sys.stdout,'reconfigure'): sys.stdout.reconfigure(encoding='utf-8')
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--predictions',type=Path,required=True)
    p.add_argument('--gold',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    rows=[json.loads(line) for line in args.predictions.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    gold=json.loads(args.gold.read_text(encoding='utf-8-sig'))
    result=evaluate(rows,gold)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open('x',encoding='utf-8') as f: json.dump(result,f,ensure_ascii=False,indent=2); f.write('\n')
    for row in result['summary']: print(row['cell'] + ': ' + format(row['official_f1_mean'],'.12f'))


if __name__=='__main__': main()

# Upstream license (verbatim):
# Attribution-NonCommercial 4.0 International
# 
# =======================================================================
# 
# Creative Commons Corporation ("Creative Commons") is not a law firm and
# does not provide legal services or legal advice. Distribution of
# Creative Commons public licenses does not create a lawyer-client or
# other relationship. Creative Commons makes its licenses and related
# information available on an "as-is" basis. Creative Commons gives no
# warranties regarding its licenses, any material licensed under their
# terms and conditions, or any related information. Creative Commons
# disclaims all liability for damages resulting from their use to the
# fullest extent possible.
# 
# Using Creative Commons Public Licenses
# 
# Creative Commons public licenses provide a standard set of terms and
# conditions that creators and other rights holders may use to share
# original works of authorship and other material subject to copyright
# and certain other rights specified in the public license below. The
# following considerations are for informational purposes only, are not
# exhaustive, and do not form part of our licenses.
# 
#      Considerations for licensors: Our public licenses are
#      intended for use by those authorized to give the public
#      permission to use material in ways otherwise restricted by
#      copyright and certain other rights. Our licenses are
#      irrevocable. Licensors should read and understand the terms
#      and conditions of the license they choose before applying it.
#      Licensors should also secure all rights necessary before
#      applying our licenses so that the public can reuse the
#      material as expected. Licensors should clearly mark any
#      material not subject to the license. This includes other CC-
#      licensed material, or material used under an exception or
#      limitation to copyright. More considerations for licensors:
#     wiki.creativecommons.org/Considerations_for_licensors
# 
#      Considerations for the public: By using one of our public
#      licenses, a licensor grants the public permission to use the
#      licensed material under specified terms and conditions. If
#      the licensor's permission is not necessary for any reason--for
#      example, because of any applicable exception or limitation to
#      copyright--then that use is not regulated by the license. Our
#      licenses grant only permissions under copyright and certain
#      other rights that a licensor has authority to grant. Use of
#      the licensed material may still be restricted for other
#      reasons, including because others have copyright or other
#      rights in the material. A licensor may make special requests,
#      such as asking that all changes be marked or described.
#      Although not required by our licenses, you are encouraged to
#      respect those requests where reasonable. More considerations
#      for the public:
#     wiki.creativecommons.org/Considerations_for_licensees
# 
# =======================================================================
# 
# Creative Commons Attribution-NonCommercial 4.0 International Public
# License
# 
# By exercising the Licensed Rights (defined below), You accept and agree
# to be bound by the terms and conditions of this Creative Commons
# Attribution-NonCommercial 4.0 International Public License ("Public
# License"). To the extent this Public License may be interpreted as a
# contract, You are granted the Licensed Rights in consideration of Your
# acceptance of these terms and conditions, and the Licensor grants You
# such rights in consideration of benefits the Licensor receives from
# making the Licensed Material available under these terms and
# conditions.
# 
# 
# Section 1 -- Definitions.
# 
#   a. Adapted Material means material subject to Copyright and Similar
#      Rights that is derived from or based upon the Licensed Material
#      and in which the Licensed Material is translated, altered,
#      arranged, transformed, or otherwise modified in a manner requiring
#      permission under the Copyright and Similar Rights held by the
#      Licensor. For purposes of this Public License, where the Licensed
#      Material is a musical work, performance, or sound recording,
#      Adapted Material is always produced where the Licensed Material is
#      synched in timed relation with a moving image.
# 
#   b. Adapter's License means the license You apply to Your Copyright
#      and Similar Rights in Your contributions to Adapted Material in
#      accordance with the terms and conditions of this Public License.
# 
#   c. Copyright and Similar Rights means copyright and/or similar rights
#      closely related to copyright including, without limitation,
#      performance, broadcast, sound recording, and Sui Generis Database
#      Rights, without regard to how the rights are labeled or
#      categorized. For purposes of this Public License, the rights
#      specified in Section 2(b)(1)-(2) are not Copyright and Similar
#      Rights.
#   d. Effective Technological Measures means those measures that, in the
#      absence of proper authority, may not be circumvented under laws
#      fulfilling obligations under Article 11 of the WIPO Copyright
#      Treaty adopted on December 20, 1996, and/or similar international
#      agreements.
# 
#   e. Exceptions and Limitations means fair use, fair dealing, and/or
#      any other exception or limitation to Copyright and Similar Rights
#      that applies to Your use of the Licensed Material.
# 
#   f. Licensed Material means the artistic or literary work, database,
#      or other material to which the Licensor applied this Public
#      License.
# 
#   g. Licensed Rights means the rights granted to You subject to the
#      terms and conditions of this Public License, which are limited to
#      all Copyright and Similar Rights that apply to Your use of the
#      Licensed Material and that the Licensor has authority to license.
# 
#   h. Licensor means the individual(s) or entity(ies) granting rights
#      under this Public License.
# 
#   i. NonCommercial means not primarily intended for or directed towards
#      commercial advantage or monetary compensation. For purposes of
#      this Public License, the exchange of the Licensed Material for
#      other material subject to Copyright and Similar Rights by digital
#      file-sharing or similar means is NonCommercial provided there is
#      no payment of monetary compensation in connection with the
#      exchange.
# 
#   j. Share means to provide material to the public by any means or
#      process that requires permission under the Licensed Rights, such
#      as reproduction, public display, public performance, distribution,
#      dissemination, communication, or importation, and to make material
#      available to the public including in ways that members of the
#      public may access the material from a place and at a time
#      individually chosen by them.
# 
#   k. Sui Generis Database Rights means rights other than copyright
#      resulting from Directive 96/9/EC of the European Parliament and of
#      the Council of 11 March 1996 on the legal protection of databases,
#      as amended and/or succeeded, as well as other essentially
#      equivalent rights anywhere in the world.
# 
#   l. You means the individual or entity exercising the Licensed Rights
#      under this Public License. Your has a corresponding meaning.
# 
# 
# Section 2 -- Scope.
# 
#   a. License grant.
# 
#        1. Subject to the terms and conditions of this Public License,
#           the Licensor hereby grants You a worldwide, royalty-free,
#           non-sublicensable, non-exclusive, irrevocable license to
#           exercise the Licensed Rights in the Licensed Material to:
# 
#             a. reproduce and Share the Licensed Material, in whole or
#                in part, for NonCommercial purposes only; and
# 
#             b. produce, reproduce, and Share Adapted Material for
#                NonCommercial purposes only.
# 
#        2. Exceptions and Limitations. For the avoidance of doubt, where
#           Exceptions and Limitations apply to Your use, this Public
#           License does not apply, and You do not need to comply with
#           its terms and conditions.
# 
#        3. Term. The term of this Public License is specified in Section
#           6(a).
# 
#        4. Media and formats; technical modifications allowed. The
#           Licensor authorizes You to exercise the Licensed Rights in
#           all media and formats whether now known or hereafter created,
#           and to make technical modifications necessary to do so. The
#           Licensor waives and/or agrees not to assert any right or
#           authority to forbid You from making technical modifications
#           necessary to exercise the Licensed Rights, including
#           technical modifications necessary to circumvent Effective
#           Technological Measures. For purposes of this Public License,
#           simply making modifications authorized by this Section 2(a)
#           (4) never produces Adapted Material.
# 
#        5. Downstream recipients.
# 
#             a. Offer from the Licensor -- Licensed Material. Every
#                recipient of the Licensed Material automatically
#                receives an offer from the Licensor to exercise the
#                Licensed Rights under the terms and conditions of this
#                Public License.
# 
#             b. No downstream restrictions. You may not offer or impose
#                any additional or different terms or conditions on, or
#                apply any Effective Technological Measures to, the
#                Licensed Material if doing so restricts exercise of the
#                Licensed Rights by any recipient of the Licensed
#                Material.
# 
#        6. No endorsement. Nothing in this Public License constitutes or
#           may be construed as permission to assert or imply that You
#           are, or that Your use of the Licensed Material is, connected
#           with, or sponsored, endorsed, or granted official status by,
#           the Licensor or others designated to receive attribution as
#           provided in Section 3(a)(1)(A)(i).
# 
#   b. Other rights.
# 
#        1. Moral rights, such as the right of integrity, are not
#           licensed under this Public License, nor are publicity,
#           privacy, and/or other similar personality rights; however, to
#           the extent possible, the Licensor waives and/or agrees not to
#           assert any such rights held by the Licensor to the limited
#           extent necessary to allow You to exercise the Licensed
#           Rights, but not otherwise.
# 
#        2. Patent and trademark rights are not licensed under this
#           Public License.
# 
#        3. To the extent possible, the Licensor waives any right to
#           collect royalties from You for the exercise of the Licensed
#           Rights, whether directly or through a collecting society
#           under any voluntary or waivable statutory or compulsory
#           licensing scheme. In all other cases the Licensor expressly
#           reserves any right to collect such royalties, including when
#           the Licensed Material is used other than for NonCommercial
#           purposes.
# 
# 
# Section 3 -- License Conditions.
# 
# Your exercise of the Licensed Rights is expressly made subject to the
# following conditions.
# 
#   a. Attribution.
# 
#        1. If You Share the Licensed Material (including in modified
#           form), You must:
# 
#             a. retain the following if it is supplied by the Licensor
#                with the Licensed Material:
# 
#                  i. identification of the creator(s) of the Licensed
#                     Material and any others designated to receive
#                     attribution, in any reasonable manner requested by
#                     the Licensor (including by pseudonym if
#                     designated);
# 
#                 ii. a copyright notice;
# 
#                iii. a notice that refers to this Public License;
# 
#                 iv. a notice that refers to the disclaimer of
#                     warranties;
# 
#                  v. a URI or hyperlink to the Licensed Material to the
#                     extent reasonably practicable;
# 
#             b. indicate if You modified the Licensed Material and
#                retain an indication of any previous modifications; and
# 
#             c. indicate the Licensed Material is licensed under this
#                Public License, and include the text of, or the URI or
#                hyperlink to, this Public License.
# 
#        2. You may satisfy the conditions in Section 3(a)(1) in any
#           reasonable manner based on the medium, means, and context in
#           which You Share the Licensed Material. For example, it may be
#           reasonable to satisfy the conditions by providing a URI or
#           hyperlink to a resource that includes the required
#           information.
# 
#        3. If requested by the Licensor, You must remove any of the
#           information required by Section 3(a)(1)(A) to the extent
#           reasonably practicable.
# 
#        4. If You Share Adapted Material You produce, the Adapter's
#           License You apply must not prevent recipients of the Adapted
#           Material from complying with this Public License.
# 
# 
# Section 4 -- Sui Generis Database Rights.
# 
# Where the Licensed Rights include Sui Generis Database Rights that
# apply to Your use of the Licensed Material:
# 
#   a. for the avoidance of doubt, Section 2(a)(1) grants You the right
#      to extract, reuse, reproduce, and Share all or a substantial
#      portion of the contents of the database for NonCommercial purposes
#      only;
# 
#   b. if You include all or a substantial portion of the database
#      contents in a database in which You have Sui Generis Database
#      Rights, then the database in which You have Sui Generis Database
#      Rights (but not its individual contents) is Adapted Material; and
# 
#   c. You must comply with the conditions in Section 3(a) if You Share
#      all or a substantial portion of the contents of the database.
# 
# For the avoidance of doubt, this Section 4 supplements and does not
# replace Your obligations under this Public License where the Licensed
# Rights include other Copyright and Similar Rights.
# 
# 
# Section 5 -- Disclaimer of Warranties and Limitation of Liability.
# 
#   a. UNLESS OTHERWISE SEPARATELY UNDERTAKEN BY THE LICENSOR, TO THE
#      EXTENT POSSIBLE, THE LICENSOR OFFERS THE LICENSED MATERIAL AS-IS
#      AND AS-AVAILABLE, AND MAKES NO REPRESENTATIONS OR WARRANTIES OF
#      ANY KIND CONCERNING THE LICENSED MATERIAL, WHETHER EXPRESS,
#      IMPLIED, STATUTORY, OR OTHER. THIS INCLUDES, WITHOUT LIMITATION,
#      WARRANTIES OF TITLE, MERCHANTABILITY, FITNESS FOR A PARTICULAR
#      PURPOSE, NON-INFRINGEMENT, ABSENCE OF LATENT OR OTHER DEFECTS,
#      ACCURACY, OR THE PRESENCE OR ABSENCE OF ERRORS, WHETHER OR NOT
#      KNOWN OR DISCOVERABLE. WHERE DISCLAIMERS OF WARRANTIES ARE NOT
#      ALLOWED IN FULL OR IN PART, THIS DISCLAIMER MAY NOT APPLY TO YOU.
# 
#   b. TO THE EXTENT POSSIBLE, IN NO EVENT WILL THE LICENSOR BE LIABLE
#      TO YOU ON ANY LEGAL THEORY (INCLUDING, WITHOUT LIMITATION,
#      NEGLIGENCE) OR OTHERWISE FOR ANY DIRECT, SPECIAL, INDIRECT,
#      INCIDENTAL, CONSEQUENTIAL, PUNITIVE, EXEMPLARY, OR OTHER LOSSES,
#      COSTS, EXPENSES, OR DAMAGES ARISING OUT OF THIS PUBLIC LICENSE OR
#      USE OF THE LICENSED MATERIAL, EVEN IF THE LICENSOR HAS BEEN
#      ADVISED OF THE POSSIBILITY OF SUCH LOSSES, COSTS, EXPENSES, OR
#      DAMAGES. WHERE A LIMITATION OF LIABILITY IS NOT ALLOWED IN FULL OR
#      IN PART, THIS LIMITATION MAY NOT APPLY TO YOU.
# 
#   c. The disclaimer of warranties and limitation of liability provided
#      above shall be interpreted in a manner that, to the extent
#      possible, most closely approximates an absolute disclaimer and
#      waiver of all liability.
# 
# 
# Section 6 -- Term and Termination.
# 
#   a. This Public License applies for the term of the Copyright and
#      Similar Rights licensed here. However, if You fail to comply with
#      this Public License, then Your rights under this Public License
#      terminate automatically.
# 
#   b. Where Your right to use the Licensed Material has terminated under
#      Section 6(a), it reinstates:
# 
#        1. automatically as of the date the violation is cured, provided
#           it is cured within 30 days of Your discovery of the
#           violation; or
# 
#        2. upon express reinstatement by the Licensor.
# 
#      For the avoidance of doubt, this Section 6(b) does not affect any
#      right the Licensor may have to seek remedies for Your violations
#      of this Public License.
# 
#   c. For the avoidance of doubt, the Licensor may also offer the
#      Licensed Material under separate terms or conditions or stop
#      distributing the Licensed Material at any time; however, doing so
#      will not terminate this Public License.
# 
#   d. Sections 1, 5, 6, 7, and 8 survive termination of this Public
#      License.
# 
# 
# Section 7 -- Other Terms and Conditions.
# 
#   a. The Licensor shall not be bound by any additional or different
#      terms or conditions communicated by You unless expressly agreed.
# 
#   b. Any arrangements, understandings, or agreements regarding the
#      Licensed Material not stated herein are separate from and
#      independent of the terms and conditions of this Public License.
# 
# 
# Section 8 -- Interpretation.
# 
#   a. For the avoidance of doubt, this Public License does not, and
#      shall not be interpreted to, reduce, limit, restrict, or impose
#      conditions on any use of the Licensed Material that could lawfully
#      be made without permission under this Public License.
# 
#   b. To the extent possible, if any provision of this Public License is
#      deemed unenforceable, it shall be automatically reformed to the
#      minimum extent necessary to make it enforceable. If the provision
#      cannot be reformed, it shall be severed from this Public License
#      without affecting the enforceability of the remaining terms and
#      conditions.
# 
#   c. No term or condition of this Public License will be waived and no
#      failure to comply consented to unless expressly agreed to by the
#      Licensor.
# 
#   d. Nothing in this Public License constitutes or may be interpreted
#      as a limitation upon, or waiver of, any privileges and immunities
#      that apply to the Licensor or You, including from the legal
#      processes of any jurisdiction or authority.
# 
# =======================================================================
# 
# Creative Commons is not a party to its public
# licenses. Notwithstanding, Creative Commons may elect to apply one of
# its public licenses to material it publishes and in those instances
# will be considered the “Licensor.” The text of the Creative Commons
# public licenses is dedicated to the public domain under the CC0 Public
# Domain Dedication. Except for the limited purpose of indicating that
# material is shared under a Creative Commons public license or as
# otherwise permitted by the Creative Commons policies published at
# creativecommons.org/policies, Creative Commons does not authorize the
# use of the trademark "Creative Commons" or any other trademark or logo
# of Creative Commons without its prior written consent including,
# without limitation, in connection with any unauthorized modifications
# to any of its public licenses or any other arrangements,
# understandings, or agreements concerning use of licensed material. For
# the avoidance of doubt, this paragraph does not form part of the
# public licenses.
# 
# Creative Commons may be contacted at creativecommons.org.
# 
