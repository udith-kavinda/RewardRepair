import json
import re
import os
import subprocess as sp

def getResults(bug_no, preds, root, bug_details):
    test_bugs = json.load(open('./bugsPHP/bug_metadata.json', 'r'))

    bug_data = bug_details[0].split('_')
    repo_owner = bug_data[0]
    repo_name = bug_data[1]
    bug_no = bug_data[3]

    print(repo_name, bug_no)

    filtered_bug = list(filter(lambda x: (x['repo_name'] == repo_name and x['bug_no'] == int(bug_no)), test_bugs))
    bug = filtered_bug[0]

    # print(bug)

    generated_bug_lines = preds

    changed_file_paths = bug['changed_file_paths']

    print(' =================================================================== ' , repo_name,  'bug_no', bug_no)
    sp.Popen(['python3', 'main.py', '-p', repo_owner+'--'+repo_name, '-b', str(bug_no), '-t', 'checkout', '-v', 'buggy', '-o', '/tmp/'], 
                cwd="/RewardRepair/bugsPHP/").communicate()

    sp.Popen(['python3', 'main.py', '-p', repo_owner+'--'+repo_name, '-b', str(bug_no), '-t', 'install', '-v', 'buggy', '-o', '/tmp/'], 
                cwd="/RewardRepair/bugsPHP/", universal_newlines=True, stdout=sp.PIPE, stderr=sp.PIPE).communicate()


    all_changing_file_lines = []

    for i in range(len(changed_file_paths)):
        path = changed_file_paths[i]
        file_path = '/tmp/' + repo_name + '/' + path

        f = open(file_path, "r")
        all_file_lines = f.readlines()
        f.close()

        all_changing_file_lines.append(all_file_lines)

        buggy_line_list = bug['changed_lines'][i]

        # generated_patch_list = patch_data['bug_data'][i]

        all_lines = all_file_lines.copy()
        line_added = 0

        for index in range(len(buggy_line_list)):
            buggy_lines = buggy_line_list[index]
            buggy_line_start = 0
            buggy_line_end = 0

            # generated_bug_lines = generated_patch_list[index]['model_generated_patches'][patches]


            # max_lines = max(len(buggy_line_list[i]['buggy']), len(buggy_line_list[i]['fixed']))
            # min_lines = min(len(buggy_line_list[i]['buggy']), len(buggy_line_list[i]['fixed']))
            if len(buggy_lines['buggy']) > 1:
                buggy_line_start = buggy_lines['buggy'][0]
                buggy_line_end = buggy_lines['buggy'][-1]
                all_lines[buggy_line_start -1: buggy_line_end] = [generated_bug_lines] + ["" for i in range(len(buggy_lines['buggy'])-1)]
                if len( buggy_lines['fixed']) > len( buggy_lines['buggy']):
                    line_added += len( buggy_lines['fixed']) - len(buggy_lines['buggy'])
                else:
                    line_added -= len( buggy_lines['buggy']) - len( buggy_lines['fixed'])

            elif len(buggy_lines['buggy']) == 1:
                buggy_line_start = buggy_lines['buggy'][0]
                buggy_line_end = buggy_lines['buggy'][0]
                # print(buggy_line_start, all_lines[buggy_line_start -1])
                # print(generated_bug_lines['replacing_patch'])
                all_lines[buggy_line_start -1] = generated_bug_lines
                if len( buggy_lines['fixed']) > len( buggy_lines['buggy']):
                    line_added += len( buggy_lines['fixed']) - len(buggy_lines['buggy'])
                else:
                    line_added -= len( buggy_lines['buggy']) - len( buggy_lines['fixed'])

            else:
                buggy_line_end = buggy_lines['fixed'][0] - line_added
                buggy_line_start = buggy_lines['fixed'][0] - line_added
                all_lines = all_lines[0:buggy_line_start-1] + [generated_bug_lines] + all_lines[buggy_line_start-1:]
                line_added += len( buggy_lines['fixed'])

            
        file = open(file_path, 'w')

        for line in all_lines:
            file.write(line)
        file.close()

        # print(index, generated_bug_lines['replacing_patch'])

    failed_test_results = sp.Popen(['python3', 'main.py', '-p', repo_owner+'--'+repo_name, '-b', str(bug_no), '-t', 'failing-test-only', '-v', 'buggy', '-o', '/tmp/'], 
                            cwd="RewardRepair/bugsPHP/", universal_newlines=True, stdout=sp.PIPE, stderr=sp.PIPE).communicate()

    failed_test_results = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?][ -/][@-~])').sub('', failed_test_results[0])

    failed_test = failed_test_results.split('\n'),

    failed_test_status = 'success' if(('ERRORS!' not in failed_test) and ('FAILURES!' not in failed_test) and ('OK' in failed_test)) else 'failed'


    all_test = ''
    if(failed_test_status == 'success'):
        all_test_result = sp.Popen(['python3', 'main.py', '-p', repo_owner+'--'+repo_name, '-b', str(bug_no), '-t', 'test', '-v', 'buggy', '-o', '/tmp/'], 
                                        cwd="RewardRepair/bugsPHP/", universal_newlines=True, stdout=sp.PIPE, stderr=sp.PIPE).communicate()
        all_test_result =  re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?][ -/][@-~])').sub('', all_test_result[0])

        all_test = all_test_result.split('\n')
    # print(all_test)

    execResult = ''
    
    if ('ERRORS!' not in failed_test) and ('FAILURES!' not in failed_test) and ('OK' not in failed_test):
        execResult = 'noTestResults'
        print('No Test Results')
           
    elif ('ERRORS!' in failed_test) or ('FAILURES!' in failed_test):
        execResult = 'failedFailingTests'
        print('Failed Failing Test Cases' ) 

    elif ('ERRORS!' not in failed_test) and ('FAILURES!' not in failed_test) and ('OK' in failed_test):
        execResult = 'passedFailingTests'
        print('Pass Failing Test Cases' )   

        # plausible
        if ('ERRORS!' not in all_test) and ('FAILURES!' not in all_test) and ('OK' in all_test):
            execResult = 'passAllTests'
            print('Plausible!!')    

    print(execResult)
    return execResult
            




if __name__ == '__main__':
    # now = datetime.datetime.now()
    # date = now.strftime("%Y-%m-%d")
    getResults('10','if  (channel  !=  null  &&  channel.getPipeline().get(HttpRequestDecoder.class)  !=  null')

