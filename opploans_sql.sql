1) Select Count(loanid) As number_of_loans, custid From all_loans Group By custid;

2)Select custid From all_loans Full Outer Join all_loanhist
On all_loans.loanid = all_loanhist.loanid Where all_loans.payoffdate >= all_loanhist.snapshot_date
Or all_loans.writeoffdate >= all_loanhist.snapshot_date
Having Count(loanid)>1;

3)Select loanid, custid, first_name, last_name, amount,
  From all_loans Where approvedate > '2019/01/01' And state = 'CA'
  And first_name In ('Matt', 'Kyle', 'Jessica', 'Mary')
  And last_name Like 'Y%';
 
4)Select Sum(amount_paid), all_loans.custid From all_loanhist Full Outer Join all_loans
On all_loans.loanid = all_loanhist.loanid Where DateDiff(snapshot_date, approvedate) = 182 Having Count(custid) <= 1;

5)Select custid, ((Sum(principal_paid) * 100) / amount) From all_loans
Full Outer Join all_loanhist On all_loans.loanid = all_loanhist.loanaid
Where DateDiff(snapshot, approvedate) = 182;
