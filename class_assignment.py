import pandas as pd
from ortools.sat.python import cp_model

def solve_class_assignment_final():
    # 1. 데이터 준비
    try:
        df = pd.read_csv('학급반편성CSP 문제 입력파일.csv')
    except FileNotFoundError:
        print("오류: '학급반편성CSP 문제 입력파일.csv' 파일을 찾을 수 없습니다.")
        return

    student_ids = df['id'].tolist()
    id_to_idx = {sid: i for i, sid in enumerate(student_ids)}
    num_students = len(df)
    
    # --- 모델 상수 ---
    NUM_CLASSES = 6
    CLASS_SIZES = [33, 33, 33, 33, 34, 34]
    ALL_STUDENTS = range(num_students)
    ALL_CLASSES = range(NUM_CLASSES)

    # 2. 모델 및 변수 생성
    model = cp_model.CpModel()

    # 핵심 변수: assignments[s][c]는 학생 s가 반 c에 배정되면 참(True)
    assignments = {}
    for s in ALL_STUDENTS:
        for c in ALL_CLASSES:
            assignments[(s, c)] = model.NewBoolVar(f'assign_s{s}_c{c}')

    # 3. Hard Constraints
    print("절대 규칙(Hard Constraints)을 적용합니다...")
    # 각 학생은 정확히 하나의 반에만 배정
    for s in ALL_STUDENTS:
        model.AddExactlyOne([assignments[(s, c)] for c in ALL_CLASSES])

    # 각 반의 학생 수는 지정된 크기를 만족
    for c in ALL_CLASSES:
        model.Add(sum(assignments[(s, c)] for s in ALL_STUDENTS) == CLASS_SIZES[c])

    # 제약조건 1A: '나쁜 관계' 학생 분리
    for _, row in df[df['나쁜관계'].notna()].iterrows():
        s1_idx = id_to_idx.get(row['id'])
        s2_idx = id_to_idx.get(int(row['나쁜관계']))
        if s1_idx is not None and s2_idx is not None:
            for c in ALL_CLASSES:
                model.AddBoolOr([assignments[(s1_idx, c)].Not(), assignments[(s2_idx, c)].Not()])

    # 제약조건 1B: '비등교' 학생은 '좋은 관계' 친구와 같은 반에 배정
    for _, row in df[(df['비등교'] == 'yes') & (df['좋은관계'].notna())].iterrows():
        s1_idx = id_to_idx.get(row['id'])
        s2_idx = id_to_idx.get(int(row['좋은관계']))
        if s1_idx is not None and s2_idx is not None:
            for c in ALL_CLASSES:
                model.Add(assignments[(s1_idx, c)] == assignments[(s2_idx, c)])

    # 제약조건 2: 각 반에는 리더십 학생이 최소 1명 이상
    leadership_students = [id_to_idx[sid] for sid in df[df['Leadership'] == 'yes']['id']]
    for c in ALL_CLASSES:
        model.Add(sum(assignments[(s, c)] for s in leadership_students) >= 1)


    # 4. Optimization Objective
    print("최적화 목표(Optimization Objective)와 패널티 시스템을 구축합니다...")

    # 핵심 목표: 성적 총점 격차 최소화
    classroom_total_scores = []
    for c in ALL_CLASSES:
        score_expr = sum(assignments[(s, c)] * df.at[s, 'score'] for s in ALL_STUDENTS)
        classroom_total_scores.append(score_expr)

    min_total_score = model.NewIntVar(0, sum(df['score']), "min_total_score")
    max_total_score = model.NewIntVar(0, sum(df['score']), "max_total_score")
    model.AddMinEquality(min_total_score, classroom_total_scores)
    model.AddMaxEquality(max_total_score, classroom_total_scores)
    
    score_range_objective = max_total_score - min_total_score

    # 추가 목표: 각종 불균형에 대한 패널티(벌점) 합산
    all_penalty_terms = []

    # 불균형에 대한 벌점을 계산하여 리스트에 추가하는 함수
    def add_balance_penalty(student_list, name):
        ideal_per_class = len(student_list) / NUM_CLASSES
        for c in ALL_CLASSES:
            count_in_class = sum(assignments[(s, c)] for s in student_list)
            deviation = model.NewIntVar(-num_students * 10, num_students * 10, f'dev_{name}_c{c}')
            model.Add(deviation == 10 * count_in_class - int(10 * ideal_per_class))
            
            abs_deviation = model.NewIntVar(0, num_students * 10, f'abs_dev_{name}_c{c}')
            model.AddAbsEquality(abs_deviation, deviation)
            all_penalty_terms.append(abs_deviation)

    # 제약조건 3, 5, 6, 7에 대한 벌점 추가
    add_balance_penalty([id_to_idx[s] for s in df[df['Piano'] == 'yes']['id']], 'piano')
    add_balance_penalty([id_to_idx[s] for s in df[df['비등교'] == 'yes']['id']], 'non_attender')
    add_balance_penalty([id_to_idx[s] for s in df[df['sex'] == 'boy']['id']], 'gender')
    add_balance_penalty([id_to_idx[s] for s in df[df['운동선호'] == 'yes']['id']], 'athletic')

    # 제약조건 8: 전년도 동급생 중복에 대한 벌점 추가
    previous_class_groups = df.groupby('24년 학급')['id'].apply(list)
    overlap_count_vars = []
    for _, members in previous_class_groups.items():
        member_indices = [id_to_idx[mid] for mid in members if mid in id_to_idx]
        for i in range(len(member_indices)):
            for j in range(i + 1, len(member_indices)):
                s1, s2 = member_indices[i], member_indices[j]
                for c in ALL_CLASSES:
                    # s1, s2 학생이 c반에서 겹치면 1이 되는 변수
                    is_overlap = model.NewBoolVar(f'overlap_{s1}_{s2}_{c}')
                    model.AddBoolAnd([assignments[(s1, c)], assignments[(s2, c)]]).OnlyEnforceIf(is_overlap)
                    overlap_count_vars.append(is_overlap)
    
    total_overlap = model.NewIntVar(0, len(overlap_count_vars), "total_overlap")
    model.Add(total_overlap == sum(overlap_count_vars))
    all_penalty_terms.append(total_overlap * 5) # 중복은 다른 것보다 5배의 벌점 부여

    # 제약조건 9: 클럽 활동 멤버 분배에 대한 벌점 추가
    club_groups = df.groupby('클럽')['id'].apply(list)
    for club_name, members in club_groups.items():
        if len(members) > 5: # 멤버가 너무 적은 클럽은 제외 (예: 5명 이하)
            member_indices = [id_to_idx[mid] for mid in members if mid in id_to_idx]
            add_balance_penalty(member_indices, f'club_{club_name}')

    # 최종 종합 목표 설정
    total_penalty = model.NewIntVar(0, 100000, "total_penalty")
    model.Add(total_penalty == sum(all_penalty_terms))
    model.Minimize(score_range_objective * 100 + total_penalty)

    # 5. 솔버 실행
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = 180.0
    solver.parameters.num_workers = 8

    status = solver.Solve(model)

    # 6. 결과 출력 및 CSV 저장
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        print(f"\n  해결책을 찾았습니다! (상태: {solver.StatusName(status)})")
        print(f"  - 성적 총점 격차 (최대-최소): {solver.Value(score_range_objective)}")
        print(f"  - 총 벌점 (불균형 점수): {solver.Value(total_penalty)}")
        print("-" * 50)
        
        # CSV 파일 저장을 위한 데이터프레임 생성
        results = []
        for s in ALL_STUDENTS:
            assigned_class = -1
            for c in ALL_CLASSES:
                if solver.Value(assignments[(s, c)]) == 1:
                    assigned_class = c + 1  # 1반부터 시작하도록 1을 더함
                    break
            
            # 원본 학생 정보와 배정된 반을 결합
            student_info = df.loc[s].copy()
            student_info['배정된_반'] = assigned_class
            results.append(student_info)

        results_df = pd.DataFrame(results)
        
        # 최종 CSV 파일 저장
        output_filename = '최종_학급반편성_결과.csv'
        results_df.to_csv(output_filename, index=False, encoding='utf-8-sig')
        print(f"\n 최종 결과가 '{output_filename}' 파일로 저장되었습니다.")
        
        class_assignments = [[] for _ in ALL_CLASSES]
        for s in ALL_STUDENTS:
            for c in ALL_CLASSES:
                if solver.Value(assignments[(s, c)]) == 1:
                    class_assignments[c].append(student_ids[s])
                    break
        
        print("\n" + "=" * 50)
        print("           반별 균형 및 구성 상세 현황 ")
        print("=" * 50)

        for c in ALL_CLASSES:
            
            class_df = df[df['id'].isin(class_assignments[c])]
            
            # 간략 정보 출력
            total_score = class_df['score'].sum()
            avg_score = class_df['score'].mean()
            gender_dist = class_df['sex'].value_counts()
            leadership_count = class_df[class_df['Leadership'] == 'yes'].shape[0]
            
            print(f"\n[ {c+1}반 편성 결과 ] (총 {len(class_assignments[c])}명)")
            print(f"  - 성적 총점: {total_score:.0f} (평균: {avg_score:.2f})")
            print(f"  - 성비: 남 {gender_dist.get('boy', 0)}명 / 여 {gender_dist.get('girl', 0)}명")
            print(f"  - 리더십: {leadership_count}명")
            
    else:
        print(f"\n 해결책을 찾지 못했습니다 (상태: {solver.StatusName(status)})")
        print("   만약 'INFEASIBLE'이라면, 절대 규칙(Hard Constraints) 간에 충돌이 있을 수 있습니다.")

# --- 프로그램 실행 ---
if __name__ == '__main__':
    solve_class_assignment_final()